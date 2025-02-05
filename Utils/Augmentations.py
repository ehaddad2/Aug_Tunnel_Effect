import torchvision.transforms as Transforms
import torch.nn as nn
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from PIL import Image, ImageFilter, ImageOps
import math

#seeding:
SEED = 30
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False

#constants
idx_to_man_aug = {
        0: "Horizontal Flip",
        1: "Resize Crop",
        2: "Random Affine",
        3: "Scale Jitter", 
        4: "Gaussian Blur",
        5: "Gaussian Noise",
        6: "Color Jitter",
        7: "Color Distortion", 
        8: "Random Invert",
        9: "Random Solarize",
        10: "Random Autocontrast",
        11: "CutOut",
        12: "Tempered MixUp",
        13: "CutMix"
}

idx_to_aug_policy = {
        0: "SwAV (2 Global + 6 Local Views)",
        1: "Barlow Twins (2 Global + 0 Local Views)", 
        2: "DINO (2 Global + 6 Local Views)"
}

"""
Helper funcs/classes
"""
class PadToSize(object):
    def __init__(self, out_H=224, out_W=224):
        self.out_H = out_H
        self.out_W = out_W

    def __call__(self, img:torch.Tensor):
        C, H, W = img.shape
        pad_H = max(0, self.out_H - H)
        pad_W = max(0, self.out_W - W)
        img = F.pad(img, (0, pad_W, 0, pad_H), value=0) #pad on bottom & right
        return img

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class AugDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform: x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.dataset)
    
class RandomApply(nn.Module): #from: https://arxiv.org/pdf/2006.07733
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
class GaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return (tensor + noise).clamp(0,1)

class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it: return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
    
def get_mean_std(dataset_name:str):
    mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.255] #defaults
    dataset_name = str.lower(dataset_name)
    if dataset_name == 'cifar-10': mean,std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]
    elif dataset_name == 'cifar-100': mean,std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    elif 'mnist' in dataset_name: mean,std = [0.1307], [0.3081]
    elif 'imagenet' in dataset_name: mean,std = [0.482, 0.458, 0.408], [0.269, 0.261, 0.276]
    return mean,std

def custom(dataset, transforms) -> Dataset:
    if not isinstance(transforms, Transforms.Compose): 
        composed = Transforms.Compose(*transforms) if len(transforms)>0 else None
    else: composed = transforms
    return AugDataset(dataset, composed)

def rand_bbox(size, lam): #based on https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
def color_distortion(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2): #based on https://arxiv.org/pdf/2006.09882
    color_jitter = Transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    rnd_color_jitter = RandomApply(color_jitter, p=0.8)
    rnd_gray = Transforms.RandomGrayscale(p=0.2)
    return [rnd_color_jitter, rnd_gray]

"""
Aug Polices
"""
class MultiCropDataset(torch.utils.data.Dataset):
    """
    Modified from SwAV: https://arxiv.org/pdf/2006.09882
    """
    def __init__(
        self,
        dataset,
        size_crops,
        nmb_crops,
        return_index=False,
        polices = [],
        out_size = (224,224)
    ):
        assert len(size_crops) == len(nmb_crops)
        assert len(polices) > 0
        self.dataset = dataset
        self.polices = polices
        self.return_index = return_index
        self.num_transforms = sum(nmb_crops) * len(polices)

        color_transform = [color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            for p in polices:
                if p == 'swav':
                    trans.extend([Transforms.Compose([
                        Transforms.RandomResizedCrop(size_crops[i], scale=([0.14, 0.05][i], [1, 0.14][i])),
                        Transforms.RandomHorizontalFlip(p=0.5),
                        Transforms.Compose(color_transform),
                        Transforms.ToTensor(),
                        Transforms.Normalize(mean=mean, std=std),
                        PadToSize(out_size[0], out_size[1])])
                    ] * nmb_crops[i])
                
                elif p == 'barlow':
                    if i == 0:  # global crops
                        trans.extend([
                            Transforms.Compose([
                                Transforms.RandomResizedCrop(224, interpolation=Transforms.InterpolationMode.BICUBIC),
                                Transforms.RandomHorizontalFlip(p=0.5),
                                Transforms.RandomApply([Transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                                Transforms.RandomGrayscale(p=0.2),
                                GaussianBlur(p=1.0),
                                Solarization(p=0.0),
                                Transforms.ToTensor(),
                                Transforms.Normalize(mean=mean, std=std),
                            ])
                        ] * nmb_crops[i])
                    else:  # local crops
                        trans.extend([
                            Transforms.Compose([
                                Transforms.RandomResizedCrop(224, interpolation=Transforms.InterpolationMode.BICUBIC),
                                Transforms.RandomHorizontalFlip(p=0.5),
                                Transforms.RandomApply([Transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                                Transforms.RandomGrayscale(p=0.2),
                                GaussianBlur(p=0.1),
                                Solarization(p=0.2),
                                Transforms.ToTensor(),
                                Transforms.Normalize(mean=mean, std=std),
                            ])
                        ] * nmb_crops[i])

                elif p == 'dino': #adopted from https://arxiv.org/pdf/2104.14294
                    flip_and_color_jitter = Transforms.Compose([
                        Transforms.RandomHorizontalFlip(p=0.5),
                        Transforms.RandomApply([
                            Transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                            Transforms.RandomGrayscale(p=0.2)])
                    if i == 0:  # first global
                        trans.extend([
                            Transforms.Compose([
                                Transforms.RandomResizedCrop(size_crops[i], scale=(0.32,1), interpolation=Transforms.InterpolationMode.BICUBIC),
                                flip_and_color_jitter,
                                GaussianBlur(p=1),
                                Transforms.ToTensor(),
                                Transforms.Normalize(mean=mean, std=std),
                            ])
                        ] * nmb_crops[i])
                    elif i==1:  # second global
                        trans.extend([
                            Transforms.Compose([
                                Transforms.RandomResizedCrop(size_crops[i], scale=(0.32,1), interpolation=Transforms.InterpolationMode.BICUBIC),
                                flip_and_color_jitter,
                                GaussianBlur(p=0.1),
                                Solarization(p=0.2),
                                Transforms.ToTensor(),
                                Transforms.Normalize(mean=mean, std=std),
                            ])
                        ] * nmb_crops[i])      

                    else: # all locals
                        trans.extend([
                            Transforms.Compose([
                                Transforms.RandomResizedCrop(size_crops[i], scale=(0.05,0.32), interpolation=Transforms.InterpolationMode.BICUBIC),
                                flip_and_color_jitter,
                                GaussianBlur(),
                                Transforms.ToTensor(),
                                Transforms.Normalize(mean=mean, std=std),
                                PadToSize(out_size[0], out_size[1])])]* nmb_crops[i])              
        self.trans = trans

    def __getitem__(self, index):
        dataset_idx = index // self.num_transforms
        transform_idx = index % self.num_transforms
        img, label = self.dataset[dataset_idx]
        transformed_img = self.trans[transform_idx](img)
        
        if self.return_index: return transformed_img, label, dataset_idx
        return transformed_img, label
    
    def __len__(self):
        return len(self.dataset) * self.num_transforms

class BarlowDataset(Dataset):
    def __init__(self, dataset, total_views=4):
        self.dataset = dataset
        self.total_views = total_views
        
        self.global_transforms = Transforms.Compose([
            RandomApply(Transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            Transforms.RandomGrayscale(p=0.2),            
            Transforms.RandomHorizontalFlip(p=0.5),
            RandomApply(Transforms.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.2),
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
        
"""
Manual Augs
"""

class ScaleJitter(object):
    """
    Perform Large Scale Jitter on the input according to 'Simple Copy-Paste is a Strong Data 
    Augmentation Method for Instance Segmentation' (https://arxiv.org/abs/2012.07177).
    """
    def __init__(self, target_size: Tuple[int, int], scale_range: Tuple[float, float] = (0.1, 2.0), interpolation: str = Transforms.InterpolationMode.BILINEAR, antialias: Optional[bool] = True):
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Randomly scaled image.
        """
        _,W,H = img.shape
        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[0] / H, self.target_size[1] / W) * scale
        new_height,new_width = int(H * r), int(W * r)
        img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode=self.interpolation, antialias=self.antialias).squeeze(0)
        pad_h = self.target_size[0] - new_height
        pad_w = self.target_size[1] - new_width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0)

class Cutout(object):
    """
    From: https://arxiv.org/abs/1708.04552
    Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
def mixup_data(x, y, alpha=1.0, n_classes=10): #from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    '''Returns mixed inputs, rand input idx's, pairs of targets, and lambda'''
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    #temper https://arxiv.org/pdf/2009.04659 
    y_onehot = torch.nn.functional.one_hot(y, num_classes=n_classes).float()
    y_index_onehot = torch.nn.functional.one_hot(y[index], num_classes=n_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_index_onehot
    y_rebalanced = abs(2 * lam - 1) * y_mixed + (1 - abs(2 * lam - 1)) / n_classes
    return mixed_x, index, y_rebalanced, lam

def cutmix_data(x, y, beta=1.0): #based on https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    '''Returns cutmixed inputs, pairs of targets, and lambda'''
    mixed_x, rand_index, y_m, lam = mixup_data(x,y, beta)
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return mixed_x, y_m, lam

def tempered_mixup_criterion(pred, y_rebalanced, lam, num_classes=10, zeta=1.0): #from https://arxiv.org/pdf/2009.04659 
    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    lam_adjusted = abs(2 * lam - 1)
    loss_confidence = -lam_adjusted * torch.sum(y_rebalanced * log_probs, dim=1)
    loss_entropy = -(1 - lam_adjusted) / num_classes * torch.sum(log_probs, dim=1)
    tempered_loss = loss_confidence.mean() + zeta * loss_entropy.mean()
    return tempered_loss
    

def get_transformations(mean, std, aug_array, img_dims = (224,224), verbose=None): 
    transformations,cutmix_b, mixup_a = [], 0, 0

    transformations.append(Transforms.Resize(256, interpolation=Transforms.InterpolationMode.BILINEAR))
    transformations.append(Transforms.CenterCrop(img_dims))
    
    # 1) Geometric transforms (PIL)
    if aug_array[0]:
        transformations.append(Transforms.RandomHorizontalFlip(p=aug_array[0]))

    if aug_array[1]: #TODO: scale and ratio fixed
        max_w, max_h, strength = img_dims[0], img_dims[1], aug_array[1]
        max_area = max_w*max_h
        crop_dim = int(math.sqrt(max_area*(1-strength+0.0001)))
        transformations.append(RandomApply(Transforms.RandomResizedCrop(size=crop_dim), p=aug_array[1]))

    if aug_array[2]:
        alpha = aug_array[2] #distribute weight evenly among param
        deg_min,deg_max = alpha*(-180), alpha*180
        scale_min, scale_max = 1-alpha+0.1, 1 + (254 * alpha)
        print(scale_min, scale_max)
        deg = alpha*90*random.choice([-1,1])
        transformations.append(RandomApply(Transforms.RandomAffine(degrees=(deg_min, deg_max), translate=(alpha, alpha), scale=(scale_min, scale_max), shear=deg), aug_array[2]))

    # 2) Color transforms (PIL)
    if aug_array[6]:
        alpha = aug_array[6]
        transformations.append(RandomApply(Transforms.ColorJitter(brightness=alpha*2.23, contrast=alpha*2.23, saturation=alpha*2.23, hue=alpha*0.5), aug_array[6]))
    if aug_array[7]:
        alpha = aug_array[7]
        t = color_distortion(brightness=alpha*2.23, contrast=alpha*2.23, saturation=alpha*2.23, hue=alpha*0.5)
        for aug in t:
            transformations.append(aug)
    if aug_array[8]:
        transformations.append(Transforms.RandomInvert(p=aug_array[8]))
    if aug_array[9]:
        alpha = aug_array[9]
        th = (1-alpha)*255
        transformations.append(Transforms.RandomSolarize(threshold=th, p=alpha))
    if aug_array[10]:
        transformations.append(Transforms.RandomAutocontrast(p=aug_array[10]))
    if aug_array[4]:
        rad_min = 0
        rad_max = 0 + aug_array[4]*(img_dims[0]/2)
        transformations.append(PILRandomGaussianBlur(radius_min=rad_min, radius_max=rad_max, p=aug_array[4]))

    mixup_a = aug_array[12]
    cutmix_b = aug_array[13]

    transformations.append(Transforms.ToTensor())
    if aug_array[3]:
        alpha = aug_array[3]
        scale_min, scale_max = 1-alpha, 1 + (img_dims[0] * alpha)
        transformations.append(RandomApply(ScaleJitter(target_size=img_dims, scale_range=(scale_min, scale_max), interpolation='bilinear'), aug_array[3]))
    
    if aug_array[5]:
        normalize = Transforms.Normalize(mean=mean, std=std)
        denormalize = Transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])
        alpha = aug_array[5]
        random_sign = random.choice([-1, 1])
        mu_upper_bound, sigma_upper_bound = 2*0.229, 3*0.229
        mu_offset = mu_upper_bound * alpha * random_sign
        sigma_offset = sigma_upper_bound * alpha * random_sign

        if random_sign==-1:
            sigma_offset = -0.229*alpha
        else: sigma_offset = ((3*0.229) - 0.229)*alpha
        transformations.append(normalize)
        transformations.append(RandomApply(GaussianNoise(mean=0.45+mu_offset, std=0.229+sigma_offset), p=aug_array[5]))
        transformations.append(denormalize)

    if aug_array[11]: 
        alpha = aug_array[11]
        cutout_area = alpha*img_dims[0]*img_dims[1]
        weight = random.random() #randomly choose weight for num holes and length tradeoff
        length = int(weight*(math.sqrt(cutout_area)))
        n_holes = int(cutout_area/(length**2))
        transformations.append(RandomApply(Cutout(n_holes, length), aug_array[11]))
    
    transformations.append(Transforms.Normalize(mean=mean, std=std))
    ret = Transforms.Compose(transformations)
    if verbose: print(f'{verbose} Manual Augmentations: {ret}\nCutmix β: {cutmix_b}\nMixup α: {mixup_a}\n')
    return ret, cutmix_b, mixup_a