import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from torchvision import transforms as T
from Utils import CustomDatasets, Augmentations, TrainTPU, TrainGPU
import Models
try:
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    from torch_xla.distributed.fsdp.wrap import always_wrap_policy
except ImportError:
    print("Note: torch_xla is not available. TPU support is disabled.")


#seeding:
SEED = 30
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

def cpu_worker(device, num_workers, dataset_base_pth, dataset_name, architecture, backbone_pth, man_aug_setting, policy_aug_setting, 
               img_dims, lr, label_smoothing, epochs, batch_size, warmup_epochs=5, use_cos_annealing=True, cuda_devices=[0]):
    """
    Worker for cpu training or if cuda available, can train DP model
    """

    train_dataset, test_dataset, num_classes = prep_data(dataset_name, img_dims, man_aug_setting, policy_aug_setting, dataset_base_pth, True)
    train_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    model = Models.Models().get_model(architecture, num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if torch.cuda.is_available(): model = nn.DataParallel(model, device_ids=cuda_devices)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    train_res = TrainGPU.train(
        model,
        train_loader,
        test_loader,
        opt,
        loss_fn,
        epochs,
        device,
        warmup_epochs,
        use_cos_annealing)
    
    Path(backbone_pth).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), backbone_pth)
    return train_res
        
        
def ddp_worker(rank, num_workers, dataset_base_pth, dataset_name, architecture, backbone_pth, man_aug_setting, 
              policy_aug_setting, img_dims, lr, label_smoothing, epochs, batch_size, ret, warmup_epochs=5, use_cos_annealing=True):
    """
    worker for cuda DDP
    """
    world_size = torch.distributed.get_world_size()
    train_dataset, test_dataset, num_classes = prep_data(dataset_name, img_dims, man_aug_setting, policy_aug_setting, dataset_base_pth, rank==0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = Models.Models().get_model(architecture, num_classes=num_classes).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(rank)
    opt = torch.optim.AdamW(ddp_model.parameters(), lr=lr, weight_decay=0.05)
    
    train_res = TrainGPU.train(
        ddp_model, 
        train_loader, 
        test_loader,
        opt, 
        loss_fn, 
        epochs, 
        rank, 
        warmup_epochs,
        use_cos_annealing)

    if rank == 0:
        Path(backbone_pth).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ddp_model.module.state_dict(), backbone_pth)
        ret.put(train_res)
    
    dist.destroy_process_group()


def tpu_worker(rank, num_workers, dataset_base_pth, dataset_name, architecture, backbone_pth, man_aug_setting, 
              policy_aug_setting, img_dims, lr, label_smoothing, epochs, batch_size, ret, warmup_epochs=5, use_cos_annealing=True):
    """
    worker for tpu/xla training
    """

    rank = xm.get_ordinal()
    train_dataset, test_dataset, num_classes = prep_data(dataset_name, img_dims, man_aug_setting, policy_aug_setting, dataset_base_pth, xm.is_master_ordinal())
    device = xm.xla_device()
    model = Models.Models().get_model(architecture, num_classes=num_classes)
    model.to(device)
    xm.broadcast_master_param(model)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=8)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size//4,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=32)

    train_loader = pl.MpDeviceLoader(
        train_loader, 
        device,
        loader_prefetch_size=512,
        device_prefetch_size=2,
        host_to_device_transfer_threads=8)
    test_loader = pl.MpDeviceLoader(
        test_loader, 
        device,
        loader_prefetch_size=128,
        device_prefetch_size=1,
        host_to_device_transfer_threads=4)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    train_res = TrainTPU.train(
        model,
        train_loader,
        test_loader,
        train_sampler,
        optimizer,
        loss_fn,
        epochs,
        device,
        warmup_epochs=warmup_epochs,
        CosAnnealing=use_cos_annealing)

    if xm.is_master_ordinal():
        Path(backbone_pth).parent.mkdir(parents=True, exist_ok=True)
        xm.save(model.state_dict(), backbone_pth)
        ret[0] = train_res

    xm.rendezvous('training_finished')

"""
-----------------|
Helper Functions |
-----------------|
"""

def prep_data(dataset_name, img_dims, man_aug_setting, policy_aug_setting, dataset_base_pth, verbose=False):
    mean, std = Augmentations.get_mean_std(dataset_name)
    test_T = Augmentations.get_transformations(mean, std, aug_array=[0] * 14, img_dims=(img_dims, img_dims), verbose="Backbone Test" if verbose else None)
    
    if not sum(policy_aug_setting):
        train_T = Augmentations.get_transformations(mean, std, aug_array=man_aug_setting, img_dims=(img_dims, img_dims), verbose="Backbone Train" if verbose else None)
        cutmix_a, mixup_a = man_aug_setting[-1], man_aug_setting[-2]
        train_dataset, test_dataset, num_classes = CustomDatasets.load_dataset(dataset_name, dataset_base_pth, train_T, test_T, cutmix_alpha=cutmix_a, mixup_alpha=mixup_a, seed=SEED, verbose=verbose)
    else:
        train_dataset, test_dataset, num_classes = CustomDatasets.load_dataset(dataset_name, dataset_base_pth, test_T, test_T, seed=SEED, verbose=verbose)
        policies = []
        if policy_aug_setting[0]:
            policies.append('swav')
            train_dataset = Augmentations.MultiCropDataset(train_dataset, [224, 96], [2, 6], policies=policies)
        if policy_aug_setting[1]:
            policies.append('barlow')
            train_dataset = Augmentations.MultiCropDataset(train_dataset, [224, 224], [1, 1], policies=policies)
        if policy_aug_setting[2]:
            policies.append('dino')
            train_dataset = Augmentations.MultiCropDataset(train_dataset, [224, 224, 96], [1, 1, 6], policies=policies)
    
    return train_dataset, test_dataset, num_classes
