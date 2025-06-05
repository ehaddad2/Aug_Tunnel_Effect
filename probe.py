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
from Utils import CustomDatasets, Augmentations
from Utils.GPU import TrainGPU
import Utils.Models as Models
from Utils.TPU import TrainTPU
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

def cpu_worker(device, num_workers, train_dataset, test_dataset, num_classes, dataset_name, backbone_ds_name, backbone_pth, backbone_arch, probe_arch, probe_layer, 
               img_dims, lr, label_smoothing, epochs, batch_size, cuda_devices=[0], save=False):
    """
    Worker for cpu training or if cuda available, can train DP model
    """
    probe = initialize_probe_model(dataset_name, num_classes, backbone_ds_name, backbone_pth, img_dims, backbone_arch, probe_arch, probe_layer)

    train_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
    
    if save:
        raise NotImplementedError("No Probe Saving Supported")
    return train_res
        
        
def ddp_worker(rank, world_size, num_workers, train_dataset, test_dataset, num_classes, dataset_name, backbone_ds_name, backbone_pth, backbone_arch, probe_arch, probe_layer, 
            img_dims, lr, label_smoothing, epochs, batch_size, ret, save=False):
    """
    worker for cuda DDP
    """
    ddp_setup(rank, world_size)
    probe = initialize_probe_model(dataset_name, num_classes, backbone_ds_name, backbone_pth, img_dims, backbone_arch, probe_arch, probe_layer).to(rank)
    ddp_model = DDP(probe, device_ids=[rank], find_unused_parameters=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=False, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=False, persistent_workers=True)
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
        if save:
            raise NotImplementedError("No Probe Saving Supported")
        ret[0] = train_res
    
    dist.destroy_process_group()


def tpu_worker(rank, num_workers, train_dataset, test_dataset, num_classes, dataset_name, backbone_ds_name, backbone_pth, backbone_arch, probe_arch, probe_layer,
                img_dims, lr, label_smoothing, epochs, batch_size, ret, save=False):
    """
    worker for tpu/xla training
    """
    
    rank = xm.get_ordinal()
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
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=20)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2)
    
    train_loader = pl.MpDeviceLoader(train_loader, device)
    test_loader = pl.MpDeviceLoader(test_loader, device)

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
        device)
    
    if xm.is_master_ordinal():
        if save:
            raise NotImplementedError("No Probe Saving Supported")
        ret[0] = train_res
    
    xm.rendezvous('training_finished')

"""
-----------------|
Helper Functions |
-----------------|
"""
def prep_data(dataset_name, img_dims, dataset_base_pth, fake=False, fake_size=1000, subset_frac=1, cache_frac=0, verbose=False):
    train_dataset, test_dataset, num_classes = None, None, 0

    if fake:
        num_classes=10
        train_dataset, test_dataset = CustomDatasets.fake_dataset(fake_size, img_dims, num_classes, SEED)
    else:
        mean, std = Augmentations.get_mean_std(dataset_name)
        T = Augmentations.get_transformations(mean, std, aug_array=[0] * 14, img_dims=(img_dims, img_dims), verbose="Probe Train/Test" if verbose else None)
        train_dataset, test_dataset, num_classes = CustomDatasets.load_dataset(dataset_name, dataset_base_pth, T, T, seed=SEED, verbose=verbose, subset_frac=subset_frac, cache_frac=cache_frac)
    return train_dataset, test_dataset, num_classes

def initialize_probe_model(dataset_name, num_classes, backbone_ds_name, backbone_pth, img_dims, backbone_arch, probe_arch, probe_layer):# TODO: support probe_arch
    out_dim = {
        'imagenet-100': 100
    }
    backbone = Models.BackboneModel().load_backbone(backbone_pth, architecture=backbone_arch, num_classes=out_dim[backbone_ds_name])
    probe = None
    if ('resnet' in backbone_arch) or ('vgg' in backbone_arch):
        probe = Models.CNNProbe(backbone, probe_layer, num_classes, img_dims)
        print(probe)
    elif 'vit' in backbone_arch:
        probe = Models.ViTHookProbe(backbone, probe_layer, num_classes)

    if not probe: raise NotImplementedError(f"Probe model not created for dataset {dataset_name} at layer {probe_layer} -- \nProbing for backbone architecture \"{backbone_arch}\" isn't supported")
    
    return probe

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)