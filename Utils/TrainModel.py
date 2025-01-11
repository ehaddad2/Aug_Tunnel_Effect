"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from torch import nn,optim
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from Utils import ManualAugs
import numpy as np
SEED = 30
def dist_training():
    return dist.is_available() and dist.is_initialized()
#seeding:
torch.manual_seed(SEED)

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
def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, ep:int, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, cutmix_beta, mixup_alpha, device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    # Create tensors to accumulate loss and accuracy
    if torch.distributed.is_initialized():
        total_loss = torch.tensor(0., device=device)
        total_acc = torch.tensor(0., device=device)
    
    if rank == 0: pbar=tqdm(total=len(dataloader), desc=f'Training Epoch {ep}')
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        """
        if cutmix_beta > 0: #perform cutmix
            X_cm, y_m, lam = ManualAugs.cutmix_data(X, y, cutmix_beta)
            outputs = model(X_cm)
            loss = ManualAugs.tempered_mixup_criterion(outputs, y_m, lam, len(dataloader.dataset.classes), zeta=1)

        elif mixup_alpha > 0: #perform mixup
            X_m, _, y_m, lam = ManualAugs.mixup_data(X, y, mixup_alpha)
            outputs = model(X_m)
            loss = ManualAugs.tempered_mixup_criterion(outputs, y_m, lam, len(dataloader.dataset.classes), zeta=1)

        elif (mixup_alpha > 0) and (cutmix_beta > 0): #perform both
            X_m, _, y_m, lam = ManualAugs.mixup_data(X, y, mixup_alpha)
            X_cm, y_m, lam = ManualAugs.cutmix_data(X_m, y_m, cutmix_beta)
            outputs = model(X_cm)
            loss = ManualAugs.tempered_mixup_criterion(outputs, y_m, lam, len(dataloader.dataset.classes), zeta=1)

        else: #no reg
            outputs = model(X)
            loss = loss_fn(outputs, y)
        """

        outputs = model(X)
        loss = loss_fn(outputs, y)
        predicted = outputs.argmax(dim=1)
        
        acc = (predicted == y).float().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        batch_acc = acc.item()
        if torch.distributed.is_initialized():
            total_loss += batch_loss
            total_acc += batch_acc
        else:
            train_loss += batch_loss
            train_acc += batch_acc
        
        if rank == 0:
            pbar.set_postfix({
                'Train Loss': f'{batch_loss:.4f}',
                'Train Acc': f'{batch_acc:.4f}'
            })
            pbar.update(1)
    
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_acc, op=torch.distributed.ReduceOp.SUM)
        train_loss = total_loss.item() / world_size
        train_acc = total_acc.item() / world_size
    
    if rank == 0: pbar.close()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(model: nn.Module, dataloader: DataLoader, ep:int, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0: pbar=tqdm(total=len(dataloader), desc=f'Testing Epoch {ep}')
    
    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(dataloader):
            #if batch_idx == 5: break
            X, y = X.to(device), y.to(device)
            
            output = model(X)
            loss = loss_fn(output, y)
            predicted = output.argmax(dim=1)
            acc = (predicted == y).float().mean()
            
            if torch.distributed.is_initialized():
                loss_tensor = torch.tensor([loss.item()], device=device)
                acc_tensor = torch.tensor([acc.item()], device=device)
                
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.SUM)
                
                batch_loss = loss_tensor.item() / world_size
                batch_acc = acc_tensor.item() / world_size
            else:
                batch_loss = loss.item()
                batch_acc = acc.item()
            test_loss += batch_loss
            test_acc += batch_acc
            
            if rank == 0:
                pbar.set_postfix({
                    'Test Loss': f'{test_loss/(batch_idx+1):.4f}',
                    'Test Acc': f'{test_acc/(batch_idx+1):.4f}'
                })
                pbar.update(1)
    
    if rank == 0: pbar.close()
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    return test_loss, test_acc

def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,  optimizer: Optimizer, loss_fn: nn.Module, epochs: int, device: torch.device, warmup_epochs: int = 0, CosAnnealing=False, cutmix_beta=0.0, mixup_alpha=0.0) -> Dict[str, List]:

    results = {"train_loss": [], "train_acc": [], 
               "test_loss": [], "test_acc": [],
               "max_test_acc": []}
    max_test_acc = 0.0
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    if warmup_epochs > 0:
        warmup_scheduler = LW_Scheduler(optimizer, warmup_epochs)
    if CosAnnealing:
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    
    for epoch in range(epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'): train_dataloader.sampler.set_epoch(epoch)
    
        train_loss, train_acc = train_step(model, train_dataloader, epoch+1, loss_fn, optimizer, cutmix_beta, mixup_alpha, device)
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        elif CosAnnealing:
            cosine_scheduler.step()
        test_loss, test_acc = test_step(model, test_dataloader, epoch+1, loss_fn, device)

        if torch.distributed.is_initialized():
            max_acc_tensor = torch.tensor([max_test_acc], device=device)
            torch.distributed.all_reduce(max_acc_tensor, op=torch.distributed.ReduceOp.MAX)
            max_test_acc = max(max_acc_tensor.item(), test_acc)
        else: max_test_acc = max(max_test_acc, test_acc)
        
        if rank == 0:
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