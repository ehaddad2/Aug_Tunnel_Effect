import sys
from pathlib import Path
import os
import torch
import torch.utils.data
from torchinfo import summary
import random
from torchmetrics import Accuracy
import torch.utils
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as Transforms
from PIL import Image
import numpy as np
import torch.nn as nn
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import collections
import argparse
from Utils import TrainModel, CustomDatasets, ManualAugs
import Models

#CONSTANTS
SEED = 30
BATCH_SIZE = 128
EPOCHS = 30
BACKBONE_OUT = 100
PROBE_LAYER = 'layer4'
PROBE_IN = 512


#seeding:
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False

class LinearProbeTrainer:
    def __init__(self, dataset_base_pth, dataset_name, num_workers, backbone_pth, probe_pth, backbone_arch, probe_arch, img_dims, probe_layer, batch_size, lr, label_smoothing, epochs, cuda_devices, device, seed, wandb, use_wandb):
        self.dataset_base_pth = dataset_base_pth
        self.dataset_name = dataset_name
        self.device = device
        self.num_workers=num_workers
        self.backbone_pth = Path(backbone_pth)
        self.backbone_arch = backbone_arch
        self.probe_pth = probe_pth
        self.probe_arch = probe_arch
        self.img_dims = img_dims
        self.probe_layer = probe_layer
        self.batch_size = batch_size
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.epochs = epochs
        self.wandb = wandb
        self.use_wandb = use_wandb
        SEED = seed
        mean, std = ManualAugs.get_mean_std(self.dataset_name)
        train_T,_,_ = ManualAugs.get_transformations(mean, std, [0]*14, img_dims=self.img_dims, verbose=f'{self.dataset_name} Probe Train')
        test_T,_,_ = ManualAugs.get_transformations(mean, std, [0]*14, img_dims=self.img_dims, verbose=f'{self.dataset_name} Probe Test')
        train, test, num_classes = CustomDatasets.load_dataset(dataset_name, self.dataset_base_pth, train_T, test_T, seed)
        self.model = self.initialize_probe_model(num_classes)
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)

    def initialize_probe_model(self, num_classes):
        models = Models.Models(self.device)
        backbone = models.get_model(self.backbone_arch, BACKBONE_OUT)
        if not Path(self.backbone_pth).exists(): raise FileNotFoundError(f"Backbone model file not found at path: {self.backbone_pth}")
        backbone.load_state_dict(torch.load(self.backbone_pth))
        for param in backbone.parameters():
            param.requires_grad = False

        probe = models.lp1(backbone=backbone, probe_layer=self.probe_layer, probe_in=PROBE_IN, probe_out=num_classes)
        return probe

    def train_probe(self, backbone_aug_setting):
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)
        self.model.to(self.device)
        train_res = TrainModel.train(
            self.model,
            self.train_loader,
            self.test_loader,
            opt,
            loss_fn,
            self.epochs,
            self.device
        )
        Path(self.probe_pth).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.probe_pth)

        """
        
        if self.use_wandb: #save to wandb

            sample_batch = next(iter(self.test_loader))[0].to(self.device)
            sample_in = sample_batch[:1]
            onnx_model = torch.onnx.export(
                self.model,
                sample_in,
                "myModel.onnx",
                input_names=["input"]
            )
            self.wandb.save(onnx_model)
        """
        return train_res
