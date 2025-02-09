"""
Contains functions for training and testing an PyTorch model on TPUs.
"""
import torch
import torch_xla.core.xla_model as xm
from torch import nn, optim
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from Utils import Augmentations
import random
import torch_xla.debug.profiler as xp

SEED = 30
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

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
def train_step(model, dataloader, ep, loss_fn, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    acc, loss, ep_acc, ep_loss, N = 0, 0, 0, 0, 0
    rank = xm.get_ordinal()
    if xm.is_master_ordinal(): pbar = tqdm(total=len(dataloader), desc=f'Training Epoch {ep}')

    for batch_idx, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        pred = outputs.argmax(dim=1)
        loss = loss_fn(outputs, y)
        ep_loss += loss.item()
        loss.backward()
        xm.optimizer_step(optimizer)
        acc = pred.eq(y.view_as(pred)).sum()
        ep_acc += acc.item()
        N += X.size()[0]
        
        if xm.is_master_ordinal():
            pbar.set_postfix({
                'Train Loss': f'{loss:.4f}',
                'Train Accuracy': f'{ep_acc/N:.4f}'})
            pbar.update(1)
    
    if xm.is_master_ordinal(): pbar.close()
    
    ep_acc = 100*ep_acc/N
    ep_acc = xm.mesh_reduce('train_ep_acc', ep_acc, np.mean)
    ep_loss = ep_loss/len(dataloader)
    return float(ep_acc), ep_loss

def test_step(model: nn.Module, dataloader: DataLoader, ep: int, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    acc, loss, ep_acc, ep_loss, N = 0, 0, 0, 0, 0
    if xm.is_master_ordinal(): pbar = tqdm(total=len(dataloader), desc=f'Testing Epoch {ep}')

    for batch_idx, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        outputs = model(X)
        pred = outputs.max(1, keepdim=True)[1]
        loss = loss_fn(outputs, y)
        ep_loss += loss.item()
        acc = pred.eq(y.view_as(pred)).sum()
        ep_acc += acc.item()
        N += X.size()[0]
        
        if xm.is_master_ordinal():
            pbar.set_postfix({
                'Test Loss': f'{loss:.4f}',
                'Test Acc': f'{ep_acc/N:.4f}'
            })
            pbar.update(1)
    if xm.is_master_ordinal(): pbar.close()
        
    ep_acc = 100*ep_acc/N
    ep_loss = ep_loss/len(dataloader)
    ep_acc = xm.mesh_reduce('test_ep_acc', ep_acc, np.mean)
    return float(ep_acc), ep_loss

def train(model, train_dataloader, test_dataloader, train_sampler, optimizer, loss_fn, epochs: int, device: torch.device, 
          warmup_epochs: int = 0, CosAnnealing=False) -> Dict[str, List]:
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
        train_sampler.set_epoch(epoch)
        train_acc, train_loss = train_step(model, train_dataloader, epoch+1, loss_fn, optimizer, device)
        test_acc, test_loss = test_step(model, test_dataloader, epoch+1, loss_fn, device)
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        elif CosAnnealing:
            cosine_scheduler.step()
        
        max_test_acc = max(max_test_acc, test_acc)
        
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