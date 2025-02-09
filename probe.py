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
except ImportError:
    pass


#seeding:
SEED = 30
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def cpu_worker(device, num_workers, dataset_base_pth, dataset_name, backbone_ds_name, backbone_pth, backbone_arch, probe_pth, probe_arch, probe_layer, 
               img_dims, lr, label_smoothing, epochs, batch_size, cuda_devices=[0]):
    """
    Worker for cpu training or if cuda available, can train DP model
    """

    train_dataset, test_dataset, num_classes = prep_data(dataset_name, img_dims, dataset_base_pth, True)
    train_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    probe = initialize_probe_model(dataset_name, num_classes, backbone_ds_name, backbone_pth, img_dims, backbone_arch, probe_arch, probe_layer, True).to(device)
    if torch.cuda.is_available(): probe = nn.DataParallel(probe, device_ids=cuda_devices)
    probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.05)
    train_res = TrainGPU.train(
        probe,
        train_loader,
        test_loader,
        opt,
        loss_fn,
        epochs,
        device)
    
    Path(probe_pth).parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), probe_pth)
    return train_res
        
        
def ddp_worker(rank, num_workers, dataset_base_pth, dataset_name, backbone_ds_name, backbone_pth, backbone_arch, probe_pth, probe_arch, probe_layer, 
            img_dims, lr, label_smoothing, epochs, batch_size, ret):
    """
    worker for cuda DDP
    """
    world_size = torch.distributed.get_world_size()
    train_dataset, test_dataset, num_classes = prep_data(dataset_name, img_dims, dataset_base_pth, rank==0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    probe = initialize_probe_model(dataset_name, num_classes, backbone_ds_name, backbone_pth, img_dims, backbone_arch, probe_arch, probe_layer, rank==0).to(rank)
    ddp_model = DDP(probe, device_ids=[rank], find_unused_parameters=True)

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
        rank)

    if rank == 0:
        Path(probe_pth).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ddp_model.module.state_dict(), probe_pth)
        ret.put(train_res)
    
    dist.destroy_process_group()


def tpu_worker(rank, num_workers, dataset_base_pth, dataset_name, backbone_ds_name, backbone_pth, backbone_arch, probe_pth, probe_arch, probe_layer,
                img_dims, lr, label_smoothing, epochs, batch_size, ret):
    """
    worker for tpu/xla training
    """
    
    rank = xm.get_ordinal()
    train_dataset, test_dataset, num_classes = prep_data(dataset_name, img_dims, dataset_base_pth, xm.is_master_ordinal())

    device = xm.xla_device()
    model = initialize_probe_model(dataset_name, num_classes, backbone_ds_name, backbone_pth, img_dims, backbone_arch, probe_arch, probe_layer).to(device)
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
        num_workers=12,
        persistent_workers=True,
        prefetch_factor=6)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        persistent_workers=True,
        prefetch_factor=2)
    
    train_loader = pl.MpDeviceLoader(train_loader, device)
    test_loader = pl.MpDeviceLoader(test_loader, device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    if xm.is_master_ordinal(): Models.print_model(model)
    train_res = TrainTPU.train(
        model,
        train_loader,
        test_loader,
        train_sampler,
        optimizer,
        loss_fn,
        epochs,
        device)
    
    if xm.is_master_ordinal():
        Path(probe_pth).parent.mkdir(parents=True, exist_ok=True)
        xm.save(model.state_dict(), probe_pth)
        ret[0] = train_res
    
    xm.rendezvous('training_finished')

"""
-----------------|
Helper Functions |
-----------------|
"""
def prep_data(dataset_name, img_dims, dataset_base_pth, verbose=False):
    mean, std = Augmentations.get_mean_std(dataset_name)
    T, _, _ = Augmentations.get_transformations(mean, std, aug_array=[0] * 14, img_dims=(img_dims, img_dims), verbose="Probe Train/Test" if verbose else None)
    train_dataset, test_dataset, num_classes = CustomDatasets.load_dataset(dataset_name, dataset_base_pth, T, T, seed=SEED, verbose=verbose)
    return train_dataset, test_dataset, num_classes

def initialize_probe_model(dataset_name, num_classes, backbone_ds_name, backbone_pth, img_dims, backbone_arch, probe_arch, probe_layer):# TODO: support probe_arch
    out_dim = {
        'imagenet-100': 100
    }

    backbone = Models.Models().get_model(architecture=backbone_arch, num_classes=out_dim[backbone_ds_name])
    backbone.load_state_dict(torch.load(backbone_pth, weights_only=True))
    for param in backbone.parameters():
        param.requires_grad = False

    probe = Models.Models().lp1(backbone=backbone, img_dims=img_dims, probe_layer=probe_layer, probe_out=num_classes)

    return probe