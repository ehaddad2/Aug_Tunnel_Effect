import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import time
from torch.multiprocessing import Queue
from torchvision import transforms as T

from Utils import TrainModel, CustomDatasets, Augmentations
import Models

SEED = 30
BATCH_SIZE = 512
EPOCHS = 100

#seeding:
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class BackboneTrainer:
    def __init__(self, dataset_base_pth, dataset_name, num_workers, architecture, backbone_pth, man_aug_setting, policy_aug_setting, img_dims, lr, label_smoothing, epochs, cuda_devices, device, seed, wandb, use_wandb, warmup_epochs=5, use_cos_annealing=True):
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
        self.wandb = wandb
        self.use_wandb = use_wandb
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
       
        models = Models.Models(device)
        self.model = models.get_model(self.architecture, num_classes=self.num_classes)

    def train_backbone_serial(self):
        if not Path.exists(Path(self.backbone_pth)):
            print("\nNo saved model found, training from scratch.")
            Path(self.backbone_pth).parent.mkdir(parents=True, exist_ok=True)
            loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)
            self.model.to(self.device)
            train_res = TrainModel.train(
                self.model,
                self.train_loader,
                self.test_loader,
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

    def main_ddp(self, world_size, batch_size):
        result_queue = Queue()
        mp.spawn(
            DDP_train, 
            args=(world_size, self.num_workers, self.model, self.train, self.test, batch_size, 
                  self.epochs, self.warmup_epochs, self.use_cos_annealing, 
                  self.label_smoothing, self.lr, self.backbone_pth, result_queue),
            nprocs=world_size, 
            join=True
        )
        return result_queue.get()

def DDP_train(rank, world_size, num_workers, model, train, test, batch_size, epochs, warmup_epochs, use_cos_annealing, label_smoothing, lr, backbone_pth, ret):
    ddp_setup(rank, world_size)
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