
"""
Contains functions for training and testing a PyTorch model on CPU and GPU (DP & DDP).
"""
import torch
from torch import nn,optim
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from Utils import Augmentations

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
def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, ep:int, loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    acc, loss, ep_acc, ep_loss, N = 0, 0, 0, 0, 0
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    cutmix_a, mixup_a = getattr(dataloader.dataset, 'cutmix_alpha', 0.0), getattr(dataloader.dataset, 'mixup_alpha', 0.0)
    
    if rank == 0: pbar=tqdm(total=len(dataloader), desc=f'Training Epoch {ep}', leave=True)
    
    for batch_idx, (X, y) in enumerate(dataloader):
        
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        """Apply cutmix/mixup"""
        if cutmix_a > 0:
            X_cm, y_1, y_2, lam = Augmentations.cutmix(X, y, cutmix_a, device)
            outputs = model(X_cm)
            loss = Augmentations.mixup_criterion(loss_fn, outputs, y_1, y_2, lam)
        elif mixup_a > 0:
            X_m, y_1, y_2, lam = Augmentations.mixup(X, y, mixup_a, device)
            outputs = model(X_m)
            loss = Augmentations.mixup_criterion(loss_fn, outputs, y_1, y_2, lam)
        elif (mixup_a > 0) and (cutmix_a > 0):
            X_m, y_1, y_2, lam = Augmentations.mixup(X, y, mixup_a, device)
            X_cm, y_c1, y_c2, lam = Augmentations.cutmix(X_m, y_1, cutmix_a, device)
            outputs = model(X_cm)
            loss = Augmentations.mixup_criterion(loss_fn, outputs, y_c1, y_c2, lam)
        else:
            outputs = model(X)
            loss = loss_fn(outputs, y)
        
        pred = outputs.argmax(dim=1)
        ep_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        acc = pred.eq(y.view_as(pred)).sum()
        ep_acc += acc.item()
        N += X.size()[0]
        
        if rank == 0:
            pbar.set_postfix({
                'Train Loss': f'{loss:.4f}',
                'Train Accuracy': f'{ep_acc/N:.4f}'})
            pbar.update(1)
    
    if rank == 0: pbar.close()

    ep_loss /= len(dataloader)
    ep_acc = ep_acc / N * 100
    
    return float(ep_acc), ep_loss

def test_step(model: nn.Module, dataloader: DataLoader, ep:int, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    acc, loss, ep_acc, ep_loss, N = 0, 0, 0, 0, 0
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank==0: pbar = tqdm(total=len(dataloader), desc=f'Testing Epoch {ep}')

    with torch.no_grad():
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
            
            if rank==0:
                pbar.set_postfix({
                    'Test Loss': f'{loss:.4f}',
                    'Test Acc': f'{ep_acc/N:.4f}'
                })
                pbar.update(1)
    if rank==0: pbar.close()
        
    ep_acc = 100*ep_acc/N
    ep_loss = ep_loss/len(dataloader)

    if torch.distributed.is_initialized(): #accum results across devices
        acc_tensor = torch.tensor([ep_acc], device=device)
        loss_tensor = torch.tensor([ep_loss], device=device)

        torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        
        world_size = torch.distributed.get_world_size()
        ep_acc = acc_tensor.item()/world_size
        ep_loss = loss_tensor.item()/world_size
    
    return float(ep_acc), float(ep_loss)

def train(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,  optimizer: Optimizer, loss_fn: nn.Module, 
          epochs: int, device: torch.device, warmup_epochs: int = 0, CosAnnealing=False) -> Dict[str, List]:

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
        if hasattr(train_dataloader.sampler, 'set_epoch'): 
            train_dataloader.sampler.set_epoch(epoch)
    
        train_acc, train_loss = train_step(model, train_dataloader, epoch+1, loss_fn, optimizer, device)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        elif CosAnnealing:
            cosine_scheduler.step()

        test_acc, test_loss = test_step(model, test_dataloader, epoch+1, loss_fn, device)

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
        setattr(train_dataloader.dataset, 'cache_ready', True)
        setattr(test_dataloader.dataset, 'cache_ready', True)

    results["max_test_acc"] = max_test_acc
    return results