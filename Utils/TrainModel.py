"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch import nn, optim
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from Utils import Augmentations
import random

SEED = 30
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False

"""
Utility Functions
"""
def LW_Scheduler(optimizer, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(warmup_epochs)
        return 1.0
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

"""
-----------------------------------------------------------------------------------------------------------------------------------------------
"""
def train_step(model, dataloader, ep:int, loss_fn, optimizer, cutmix_beta, mixup_alpha, device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0.0, 0.0
    
    # Only show progress bar on master process
    if xm.is_master_ordinal():
        pbar = tqdm(total=len(dataloader), desc=f'Training Epoch {ep}')
    
    for batch_idx, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X = X.to(xm.xla_device())
        y = y.to(xm.xla_device())
        if cutmix_beta:
            X_cm, y_m, lam = Augmentations.cutmix_data(X, y, cutmix_beta)
            outputs = model(X_cm)
            #loss = ManualAugs.tempered_mixup_criterion(outputs, y_m, lam, len(dataloader.dataset.classes), zeta=1)
        elif mixup_alpha:
            X_m, _, y_m, lam = Augmentations.mixup_data(X, y, mixup_alpha)
            outputs = model(X_m)
            #loss = ManualAugs.tempered_mixup_criterion(outputs, y_m, lam, len(dataloader.dataset.classes), zeta=1)
        elif mixup_alpha and cutmix_beta:
            X_m, _, y_m, lam = Augmentations.mixup_data(X, y, mixup_alpha)
            X_cm, y_m, lam = Augmentations.cutmix_data(X_m, y_m, cutmix_beta)
            outputs = model(X_cm)
            #loss = ManualAugs.tempered_mixup_criterion(outputs, y_m, lam, len(dataloader.dataset.classes), zeta=1)
        else:
            outputs = model(X)
            
        loss = loss_fn(outputs, y)
        loss.backward()
        
        # Use XLA optimizer step
        xm.optimizer_step(optimizer)
        xm.mark_step()
        
        predicted = outputs.argmax(dim=1)
        acc = (predicted == y).float().mean()
        
        batch_loss = loss.item()
        batch_acc = acc.item()
        
        train_loss += batch_loss
        train_acc += batch_acc
        
        if xm.is_master_ordinal():
            pbar.set_postfix({
                'Train Loss': f'{batch_loss:.4f}',
                'Train Accuracy': f'{batch_acc:.4f}'
            })
            pbar.update(1)
    
    if xm.is_master_ordinal():
        pbar.close()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(model: nn.Module, dataloader: DataLoader, ep: int, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    
    if xm.is_master_ordinal():
        pbar = tqdm(total=len(dataloader), desc=f'Testing Epoch {ep}')
    
    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(dataloader):
            X = X.to(xm.xla_device())
            y = y.to(xm.xla_device())
            output = model(X)
            loss = loss_fn(output, y)
            predicted = output.argmax(dim=1)
            acc = (predicted == y).float().mean()
            
            batch_loss = loss.item()
            batch_acc = acc.item()
            
            test_loss += batch_loss
            test_acc += batch_acc
            
            # Mark step for XLA
            xm.mark_step()
            
            if xm.is_master_ordinal():
                pbar.set_postfix({
                    'Test Loss': f'{test_loss/(batch_idx+1):.4f}',
                    'Test Acc': f'{test_acc/(batch_idx+1):.4f}'
                })
                pbar.update(1)
    
    if xm.is_master_ordinal():
        pbar.close()
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    return test_loss, test_acc

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs: int, device: torch.device, 
          warmup_epochs: int = 0, CosAnnealing=False, cutmix_beta=0.0, mixup_alpha=0.0) -> Dict[str, List]:
    results = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "max_test_acc": []
    }
    max_test_acc = 0.0
    
    if warmup_epochs > 0:
        warmup_scheduler = LW_Scheduler(optimizer, warmup_epochs)
    if CosAnnealing:
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, epoch+1, loss_fn, optimizer, cutmix_beta, mixup_alpha, device)
        test_loss, test_acc = test_step(model, test_dataloader, epoch+1, loss_fn, device)
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        elif CosAnnealing:
            cosine_scheduler.step()
        
        # Update max accuracy
        max_test_acc = max(max_test_acc, test_acc)
        
        # Only master process updates and prints results
        if xm.is_master_ordinal():
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            print(f"Epoch: {epoch+1} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | "
                  f"test_acc: {test_acc:.4f} | "
                  f"max_test_acc: {max_test_acc:.4f}")
    
    results["max_test_acc"] = max_test_acc
    return results