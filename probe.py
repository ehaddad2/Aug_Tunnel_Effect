import sys
sys.path.append('/home/elias/Deep Learning/Utils')
import os
import torch
import torch.utils.data
from torchinfo import summary
import random
from torchmetrics import Accuracy
import torch.utils
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
from torchvision import datasets
import torchvision.transforms as Transforms
from PIL import Image
import numpy as np
import torch.nn as nn
from timeit import default_timer as timer
import TrainModel
import matplotlib.pyplot as plt
import collections
import CustomDatasets

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="/home/elias/Deep Learning/Research/OOD/runs")

#CONSTANTS
SEED = 30
BATCH_SIZE = 128
EPOCHS = 30
DATASET_NAME = "imagenet-100" 
DATASET_BASE_PATH = "/home/elias/Deep Learning/Research/OOD/data/"
BACKBONE_PATH = "/home/elias/Deep Learning/Research/OOD/models/IN-100_Test3/res18_0.pth"
PROBE_PATH = "/home/elias/Deep Learning/Research/OOD/models/Probes/IN-100_Test3/IN-100/res18_0.pth"
BACKBONE_OUT = 100
PROBE_LAYER = 'layer4'
PROBE_IN = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'\nDevice being used: ', device, '\n')

#seeding:
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False

"""
Classes
"""
class DatasetClone(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform: x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.dataset)
    
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0, std=0.01)
        self.linear.bias.data.zero_()
        

    def forward(self, x):
        return self.linear(x)   

"""
Functions
"""
def print_model(model:nn.Module):
    print(model)
    print("\nTrainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad: print(f"{name} (trainable)")       
        else: print(f"{name} (frozen)")

def get_mean_std(dataset_name:str):
    mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.255] #defaults
    dataset_name = str.lower(dataset_name)
    if dataset_name == 'cifar-10': mean,std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]
    elif dataset_name == 'cifar-100': mean,std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif 'mnist' in dataset_name: mean,std = [0.1307], [0.3081]
    elif 'imagenet' in dataset_name: mean,std = [0.482, 0.458, 0.408], [0.269, 0.261, 0.276]
    return mean,std

def linear_probe(backbone:nn.Module, probe_layer:str, probe_in:int, probe_out:int): #performs 1x1 adaptive avg pooling then attaches linear head after probe_layer
    layers = collections.OrderedDict()
    for name, layer in backbone.named_children():
        layers[name] = layer
        if name == probe_layer:break

    probe = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        LinearClassifier(probe_in, probe_out)
    )
    layers['probe'] = probe
    new_backbone = nn.Sequential(layers)
    return new_backbone

def load_dataset(dataset_name:str, transforms:Transforms.Compose): #loads in a dataset with initial transoformations
    assert(dataset_name)
    dataset_name = str.lower(dataset_name)
    train,test,num_classes = None,None,0

    if dataset_name == 'cifar-10': 
        train,test = datasets.CIFAR10(root=DATASET_BASE_PATH+'cifar-10', transform=transforms, download=True), datasets.CIFAR10(root=DATASET_BASE_PATH+'cifar-10', train=False, transform=transforms, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'cifar-100': 
        train,test = datasets.CIFAR100(root=DATASET_BASE_PATH+'cifar-100', transform=transforms, download=True), datasets.CIFAR100(root=DATASET_BASE_PATH+'cifar-100', train=False, transform=transforms, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'flowers-102': 
        train,test = datasets.Flowers102(root=DATASET_BASE_PATH, split='test', transform=transforms, download=True), torch.utils.data.ConcatDataset([datasets.Flowers102(root=DATASET_BASE_PATH+'flowers-102', split='train', transform=transforms, download=True), datasets.Flowers102(root=DATASET_BASE_PATH+'flowers-102', split='val', transform=transforms, download=True)])
        num_classes = 102
    elif dataset_name == 'stl-10': 
        train,test = datasets.STL10(root=DATASET_BASE_PATH+'stl-10', split='test', transform=transforms, download=True), datasets.STL10(root=DATASET_BASE_PATH+'stl-10', split='train', transform=transforms, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'aircrafts': 
        train,test = datasets.FGVCAircraft(root=DATASET_BASE_PATH+'aircrafts', split='train', transform=transforms, download=True), datasets.FGVCAircraft(root=DATASET_BASE_PATH+'aircrafts', split='test', transform=transforms, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'cub-200': 
        train,test = CustomDatasets.Cub2011(root=DATASET_BASE_PATH+'cub-200', transform=transforms, download=True), CustomDatasets.Cub2011(root=DATASET_BASE_PATH+'cub-200', train=False, transform=transforms, download=True)
        num_classes = 200
    else: 
        dataset = torchvision.datasets.ImageFolder(DATASET_BASE_PATH+dataset_name)
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = len(dataset.classes)
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(SEED)) 
        train,test = DatasetClone(train, transform=transforms), DatasetClone(test, transform=transforms)

    print('\ntrain length: ', len(train), 'test length: ', len(test), '\n')
    return train,test,num_classes

def main():
    """
    Data
    """
    mean,std = get_mean_std(DATASET_NAME)
    transforms0 = Transforms.Compose([
        Transforms.Resize(256, interpolation=Transforms.InterpolationMode.BILINEAR),
        Transforms.CenterCrop(224),
        Transforms.ToTensor(),
        Transforms.Normalize(mean=mean, std=std)])

    train,test,num_classes = load_dataset(DATASET_NAME, Transforms.Compose([transforms0]))
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    img, label = next(iter(train_dataloader))
    print(f'\nBatch dimensions:', img.shape, '\n')

    """
    Models
    """
    backbone = torchvision.models.resnet18()
    backbone.fc = nn.Linear(512, BACKBONE_OUT) #512 changes if we use different backbone
    backbone.load_state_dict(torch.load(BACKBONE_PATH))

    for param in backbone.parameters():
        param.requires_grad = False

    model = linear_probe(backbone=backbone, probe_layer=PROBE_LAYER, probe_in=PROBE_IN, probe_out=num_classes)
    model.to(device)
    #summary(model, input_size=img.shape, col_names=["trainable", "input_size", "output_size", "kernel_size"])

    """
    Train/Test
    """
    img, label = next(iter(train_dataloader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    results = TrainModel.train(model, train_dataloader, test_dataloader, opt, loss_fn, EPOCHS, device, writer, cuda_devices=[0])
    torch.save(model.state_dict(), PROBE_PATH)

if __name__ == '__main__':
    main()