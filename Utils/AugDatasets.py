import torch
import torch.nn as nn
import random
import math
import sys
from torch.utils.data import Dataset, Subset, ConcatDataset
import torchvision.transforms as Transforms
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
sys.path.append("./Utils")
from Utils import ManualAugs

SEED = 30

#seeding:
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

"""
Functions
"""

class SwavDataset(Dataset):
    def __init__(self, dataset, V=4, G=2):
        self.dataset = dataset
        self.V = V
        self.G = G
        def get_color_distortion(s=1.0): #function adopted from https://arxiv.org/pdf/2006.09882
        # s is the strength of color distortion.
            color_jitter = Transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            rnd_color_jitter = Transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = Transforms.RandomGrayscale(p=0.2)
            color_distort = Transforms.Compose([rnd_color_jitter, rnd_gray])
            return color_distort
        
        self.local_transforms = Transforms.Compose([
            Transforms.RandomResizedCrop(96, scale=(0.05, 0.14)),
            Transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(),
            ManualAugs.PILRandomGaussianBlur(),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
        ])
        self.global_transforms = Transforms.Compose([
            Transforms.RandomResizedCrop(224, scale=(0.14, 1)),
            Transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(),
            ManualAugs.PILRandomGaussianBlur(),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
        ])
        self.total_length = len(self.dataset) * (self.G + self.V)
    
    def __getitem__(self, idx):
        dataset_idx = idx // (self.V + self.G)
        transform_idx = idx % (self.V + self.G)
        sample, label = self.dataset[dataset_idx]
        if transform_idx < self.V:
            sample = self.local_transforms(sample)
        else:
            sample = self.global_transforms(sample)
        return sample, label
    
    def __len__(self):
        return self.total_length
    

class BarlowDataset(Dataset):
    def __init__(self, dataset, total_views=4):
        self.dataset = dataset
        self.total_views = total_views
        
        self.global_transforms = Transforms.Compose([
            ManualAugs.RandomApply(Transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            Transforms.RandomGrayscale(p=0.2),            
            Transforms.RandomHorizontalFlip(p=0.5),
            ManualAugs.RandomApply(Transforms.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.2),
            Transforms.RandomResizedCrop(224, scale=(0.08, 1), interpolation=Transforms.InterpolationMode.BICUBIC),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
        ])
        self.total_length = len(self.dataset) * self.total_views
    
    def __getitem__(self, idx):
        dataset_idx = idx // (self.total_views)
        sample, label = self.dataset[dataset_idx]
        sample = self.global_transforms(sample)
        return sample, label
    
    def __len__(self):
        return self.total_length