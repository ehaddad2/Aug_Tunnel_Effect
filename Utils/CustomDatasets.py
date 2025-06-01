import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision.transforms as Transforms
from torchvision import datasets
import torch
import torchvision
import torchaudio.transforms as transforms
from tqdm import tqdm
from PIL import Image
import librosa
from torch.utils.data import Dataset
import math

def identity_collate(batch):
    return batch

class ManualAugDataset(Dataset):
    def __init__(self, dataset, transform=None, cutmix_alpha=0.0, mixup_alpha=0.0, num_classes=10, subset_frac=1, cache_frac=0):
        subset_size = int(subset_frac * len(dataset))
        self.indices = list(range(subset_size))
        self.dataset = Subset(dataset, self.indices)
        self.transform = transform
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.cache_size = int(cache_frac * len(self.dataset))  # Convert to int here
        self.cache_data = None
        self.cache_labels = None

        if self.cache_size > 0:  # Store samples in RAM cache of specified size
            self.cache_data = [None] * self.cache_size
            self.cache_labels = [None] * self.cache_size

            print(f"Preloading {cache_frac*100:.1f}% of dataset to cache...")
            loader = DataLoader(
                self.dataset,
                batch_size=256,
                num_workers=12,
                pin_memory=False,
                collate_fn=identity_collate
            )

            idx = 0
            for batch in tqdm(loader, total=math.ceil(self.cache_size/256)-1, desc="Caching"):
                for x, y in batch:
                    if idx >= self.cache_size:  # Fixed condition
                        break
                    if self.transform:
                        x = self.transform(x)
                    self.cache_data[idx] = x
                    self.cache_labels[idx] = torch.tensor(y)
                    idx += 1
                
                if idx >= self.cache_size:  # Break outer loop too
                    break
            
            print("Preloading complete.")

    def __getitem__(self, index):
        if self.cache_size > 0 and index < self.cache_size:  # Check if index is within cache
            return self.cache_data[index], self.cache_labels[index]
        
        # Fallback to original dataset
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, torch.tensor(y)
    
    def __len__(self):
        return len(self.dataset)
    

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, metadata, audio_dir, transform=None, sample_rate=22050):
        self.data = []
        self.audio_dir = audio_dir
        self.transform = transform
        self.sample_rate = sample_rate
        
        with open(metadata, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                fields = line.strip().split(',')
                file_path = os.path.join(audio_dir, fields[0])
                label = int(fields[2])
                self.data.append((file_path, label))
    
    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        waveform = torch.from_numpy(waveform).float()
        waveform = waveform.unsqueeze(0) 
        if self.transform:
            waveform = Transforms.ToPILImage()(self.transform(waveform).repeat(3, 1, 1))
        return waveform, label
    
    def __len__(self):
        return len(self.data)
    
class HAM10000Dataset(Dataset):
    def __init__(self, metadata, img_dirs, transform=None):
        """
        Args:
            metadata (str): Path to the metadata CSV file.
            img_dirs (list): List of directories containing images (e.g., Part 1 and Part 2).
            transform (callable, optional): Transformations to apply to the images.
        """
        self.data = pd.read_csv(metadata)
        self.img_dirs = img_dirs
        self.transform = transform
        self.label_dict = {
            'nv'   : 0,
            'mel'  : 1,
            'bkl'  : 2,
            'bcc'  : 3,
            'akiec': 4,
            'df'   : 5,
            'vasc' : 6
        }

    def __getitem__(self, idx): #pair csv label to img path
        img_name = self.data.iloc[idx]['image_id'] + ".jpg"
        dx_str = self.data.iloc[idx]['dx']
        label = torch.tensor(self.label_dict[dx_str], dtype=torch.long)
        img_path = None
        for img_dir in self.img_dirs:
            potential_path = os.path.join(img_dir, img_name)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in provided directories.")

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.data)
    

def custom_dataset(dataset, transforms, num_classes, cutmix_alpha=0, mixup_alpha=0, subset_frac=1, cache_frac=0) -> Dataset:
    if not isinstance(transforms, Transforms.Compose): 
        composed = Transforms.Compose(*transforms) if len(transforms)>0 else None
    else: composed = transforms
    return ManualAugDataset(dataset, composed, cutmix_alpha, mixup_alpha, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)

def fake_dataset(size, img_dims, num_classes, seed=30):
    ds = TensorDataset(torch.rand(size, 3, img_dims, img_dims), torch.randint(0, num_classes, (size,)))
    lengths = [int(len(ds)*0.8), len(ds) - int(len(ds)*0.8)]
    train, test = torch.utils.data.random_split(ds, lengths, generator=torch.Generator().manual_seed(seed)) 
    return train, test

def load_dataset(dataset_name, base_pth, train_T = [], test_T = [], cutmix_alpha=0, mixup_alpha=0, seed = None, verbose=False, subset_frac=1, cache_frac=0): #loads in a dataset with initial transoformations
    dataset_name = str.lower(dataset_name)
    train,test,num_classes = None,None,0

    if dataset_name == 'cifar-10': 
        train,test = datasets.CIFAR10(root=base_pth+'cifar-10', download=True), datasets.CIFAR10(root=base_pth+'cifar-10', train=False, download=True)
        num_classes = len(train.classes)
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'cifar-100': 
        train,test = datasets.CIFAR100(root=base_pth+'cifar-100', download=True), datasets.CIFAR100(root=base_pth+'cifar-100', train=False, download=True)
        num_classes = len(train.classes)
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'flowers-102': 
        test,train = datasets.Flowers102(root=base_pth, split='test', download=True), torch.utils.data.ConcatDataset([datasets.Flowers102(root=base_pth+'flowers-102', split='train', download=True), datasets.Flowers102(root=base_pth+'flowers-102', split='val', download=True)])
        num_classes = 102
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'stl-10': 
        test,train = datasets.STL10(root=base_pth+'stl-10', split='test', download=True), datasets.STL10(root=base_pth+'stl-10', split='train', download=True)
        num_classes = len(train.classes)
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'aircrafts': 
        train,test = datasets.FGVCAircraft(root=base_pth+'aircrafts', split='train', download=True), datasets.FGVCAircraft(root=base_pth+'aircrafts', split='test', download=True)
        num_classes = len(train.classes)
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'cub-200': 
        train,test = Cub2011(root=base_pth+'cub-200', download=True), Cub2011(root=base_pth+'cub-200', train=False, download=True)
        num_classes = 200
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'ninco': 
        dataset = torchvision.datasets.ImageFolder(base_pth+dataset_name)
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = len(dataset.classes)
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)) 
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'oxford-pets':
        test,train = datasets.OxfordIIITPet(root=base_pth+'oxford-pets', split='trainval', download=True), Cub2011(root=base_pth+'oxford-pets', split='train', download=True)
        num_classes = 200
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'ham10000':
        ds1, ds2 = base_pth+dataset_name+'/p1', base_pth+dataset_name+'/p2'
        dataset = HAM10000Dataset(base_pth+dataset_name+'/metadata.csv',[ds1,ds2])
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = 7
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)) 
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    elif dataset_name == 'esc-50':
        mel_transform = transforms.MelSpectrogram(sample_rate=44100, n_fft=2205,hop_length=441)
        ds_pth = base_pth+dataset_name
        dataset = ESC50Dataset(ds_pth+'/metadata.csv', ds_pth+'/audio', transform=mel_transform)
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = 50
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)) 
        train,test = custom_dataset(train, train_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)
    else: 
        train, test = torchvision.datasets.ImageFolder(base_pth+dataset_name+'/train/'), torchvision.datasets.ImageFolder(base_pth+dataset_name+'/val/')
        num_classes = len(train.classes)
        train,test = custom_dataset(train, train_T, num_classes, cutmix_alpha, mixup_alpha, subset_frac=subset_frac, cache_frac=cache_frac), custom_dataset(test, test_T, num_classes, subset_frac=subset_frac, cache_frac=cache_frac)

    if verbose: print('\ntrain length: ', len(train), 'test length: ', len(test), '\n')
    return train,test,num_classes