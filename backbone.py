import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import time
from torch.multiprocessing import Queue
from torchvision import transforms as T
from Utils import TrainModel, CustomDatasets, Augmentations
import Models
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torchvision
import torchvision.transforms as transforms
import torch_xla.distributed.spmd as xs
#seeding:
SEED = 30
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False

class BackboneTrainer:
    def __init__(self, dataset_base_pth, dataset_name, num_workers, architecture, backbone_pth, man_aug_setting, policy_aug_setting, img_dims, lr, label_smoothing, epochs, cuda_devices, use_tpu, device, seed, wandb, use_wandb, warmup_epochs=5, use_cos_annealing=True):
        self.dataset_base_pth = dataset_base_pth
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.architecture = architecture
        self.device = device
        self.backbone_pth = Path(backbone_pth)
        self.man_aug_setting = man_aug_setting
        self.policy_aug_setting = policy_aug_setting
        self.img_dims = img_dims
        self.epochs = epochs
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.warmup_epochs = warmup_epochs
        self.use_cos_annealing = use_cos_annealing
        self.cuda_devices = cuda_devices
        self.use_tpu = use_tpu
        self.wandb = wandb
        self.use_wandb = use_wandb
        self.cuda_devices = cuda_devices
        SEED = seed

        mean, std = Augmentations.get_mean_std(dataset_name)
        test_T,_,_ = Augmentations.get_transformations(mean, std, aug_array=[0]*14, img_dims=self.img_dims, verbose="Backbone Test") # manual augs
        if not sum(self.policy_aug_setting):
            train_T,self.cutmix_b,self.mixup_a = Augmentations.get_transformations(mean, std, aug_array=man_aug_setting, img_dims=self.img_dims, verbose="Backbone Train")
            self.train, self.test, self.num_classes = CustomDatasets.load_dataset(dataset_name, self.dataset_base_pth, train_T, test_T, seed)
        else:
            self.train, self.test, self.num_classes = CustomDatasets.load_dataset(dataset_name, self.dataset_base_pth, T.Compose([]), test_T, seed)
            polices = []
            if self.policy_aug_setting[0]:
                polices.append('swav')
                self.train = Augmentations.MultiCropDataset(self.train, [224,96], [2,6], polices=polices)
            
            if self.policy_aug_setting[1]:
                polices.append('barlow')
                self.train = Augmentations.MultiCropDataset(self.train, [224,224], [1,1], polices=polices)
            
            if self.policy_aug_setting[2]:
                polices.append('dino')
                self.train = Augmentations.MultiCropDataset(self.train, [224,224,96], [1,1,6], polices=polices)
       
        models = Models.Models()
        self.model = models.get_model(self.architecture, num_classes=self.num_classes)

    def cpu_train(self, batch_size):
        if not Path.exists(Path(self.backbone_pth)):
            print("\nNo saved model found, training from scratch.")
            train_loader = DataLoader(self.train, batch_size, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(self.test, batch_size, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
            Path(self.backbone_pth).parent.mkdir(parents=True, exist_ok=True)
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)
            if torch.cuda.is_available(): self.model = nn.DataParallel(self.model, device_ids=self.cuda_devices)
            self.model.to(self.device)
            train_res = TrainModel.train(
                self.model,
                train_loader,
                test_loader,
                opt,
                loss_fn,
                self.epochs,
                self.device,
                self.warmup_epochs,
                self.use_cos_annealing,
                self.cutmix_b,
                self.mixup_a)
            
            torch.save(self.model.state_dict(), self.backbone_pth)
            return train_res
        else:
            print("\nSaved model found, loading it.")
            self.model.load_state_dict(torch.load(self.backbone_pth))

    def ddp_train(self, batch_size):
        result_queue = Queue()
        
        mp.spawn( #transfer workers to GPUs
            ddp_worker, 
            args=(len(self.cuda_devices), self.num_workers, self.model, self.train, self.test, batch_size, 
                  self.epochs, self.warmup_epochs, self.use_cos_annealing, 
                  self.label_smoothing, self.lr, self.backbone_pth, result_queue),
            nprocs=len(self.cuda_devices), 
            join=True
        )
        return result_queue.get()
        
        
def ddp_worker(rank, world_size, num_workers, model, train, test, batch_size, epochs, warmup_epochs, use_cos_annealing, label_smoothing, lr, backbone_pth, ret):
    """
    only use with cuda
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    train_sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    test_sampler = DistributedSampler(test, num_replicas=world_size, rank=rank, shuffle=False, seed=SEED)
    train_loader = DataLoader(train, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(rank)
    opt = torch.optim.AdamW(ddp_model.parameters(), lr=lr, weight_decay=0.05)
    
    train_res = TrainModel.train(
        ddp_model, 
        train_loader, 
        test_loader, 
        opt, 
        loss_fn, 
        epochs, 
        rank, 
        warmup_epochs=warmup_epochs,
        CosAnnealing=use_cos_annealing
    )

    if rank == 0:
        Path(backbone_pth).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ddp_model.module.state_dict(), backbone_pth)
        ret.put(train_res)
    
    dist.destroy_process_group()


def tpu_worker(rank, num_workers, dataset_base_pth, dataset_name, architecture, backbone_pth, man_aug_setting, 
              policy_aug_setting, img_dims, lr, label_smoothing, epochs, batch_size, warmup_epochs=5, use_cos_annealing=True):
    """
    Standalone XLA training function that uses XLA's built-in replication.
    """
    # Data & Augmentations
    mean, std = Augmentations.get_mean_std(dataset_name)
    test_T, _, _ = Augmentations.get_transformations(mean, std, aug_array=[0] * 14, img_dims=img_dims, verbose="Backbone Test")
    
    if not sum(policy_aug_setting):
        train_T, cutmix_b, mixup_a = Augmentations.get_transformations(mean, std, aug_array=man_aug_setting, img_dims=img_dims, verbose="Backbone Train")
        train_dataset, test_dataset, num_classes = CustomDatasets.load_dataset(dataset_name, dataset_base_pth, train_T, test_T, SEED)
    else:
        train_dataset, test_dataset, num_classes = CustomDatasets.load_dataset(dataset_name, dataset_base_pth, T.Compose([]), test_T, SEED)
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

    # Get XLA device
    device = xm.xla_device()
    
    # Create model and move to XLA device
    model = Models.Models().get_model(architecture, num_classes=num_classes).to(device)
    
    # Broadcast master parameters to all TPU cores
    xm.broadcast_master_param(model)
    
    # Create DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=32
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=32
    )
    
    # Wrap with MpDeviceLoader for better TPU performance
    train_loader = pl.MpDeviceLoader(train_loader, device, loader_prefetch_size=128, device_prefetch_size=1, host_to_device_transfer_threads=4)
    test_loader = pl.MpDeviceLoader(test_loader, device, loader_prefetch_size=128, device_prefetch_size=1, host_to_device_transfer_threads=4)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    # Training
    train_res = TrainModel.train(
        model,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        epochs,
        device,
        warmup_epochs=warmup_epochs,
        CosAnnealing=use_cos_annealing,
        cutmix_beta=cutmix_b if 'cutmix_b' in locals() else 0.0,
        mixup_alpha=mixup_a if 'mixup_a' in locals() else 0.0
    )
    
    # Save model only from master process
    if xm.is_master_ordinal():
        Path(backbone_pth).parent.mkdir(parents=True, exist_ok=True)
        xm.save(model.state_dict(), backbone_pth)
    
    # Ensure all cores are synced before finishing
    xm.rendezvous('training_finished')
    
    return train_res if xm.is_master_ordinal() else None