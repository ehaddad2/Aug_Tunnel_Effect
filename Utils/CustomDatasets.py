import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
from torchvision import datasets
import torch
import torchvision
import torchaudio
import torchaudio.transforms as transforms
import librosa
from Utils import ManualAugs
from PIL import Image


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
        waveform, sr = torchaudio.load(file_path)
 
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
    
def load_dataset(dataset_name:str, base_pth, train_T, test_T, seed): #loads in a dataset with initial transoformations
    assert(dataset_name)
    dataset_name = str.lower(dataset_name)
    train,test,num_classes = None,None,0

    if dataset_name == 'cifar-10': 
        train,test = datasets.CIFAR10(root=base_pth+'cifar-10',transform=train_T, download=True), datasets.CIFAR10(root=base_pth+'cifar-10', train=False, transform=test_T, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'cifar-100': 
        train,test = datasets.CIFAR100(root=base_pth+'cifar-100', transform=train_T, download=True), datasets.CIFAR100(root=base_pth+'cifar-100', train=False, transform=test_T, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'flowers-102': 
        train,test = datasets.Flowers102(root=base_pth, split='test', transform=test_T, download=True), torch.utils.data.ConcatDataset([datasets.Flowers102(root=base_pth+'flowers-102', split='train', transform=train_T, download=True), datasets.Flowers102(root=base_pth+'flowers-102', split='val', transform=train_T, download=True)])
        num_classes = 102
    elif dataset_name == 'stl-10': 
        train,test = datasets.STL10(root=base_pth+'stl-10', split='test', transform=test_T, download=True), datasets.STL10(root=base_pth+'stl-10', split='train', transform=train_T, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'aircrafts': 
        train,test = datasets.FGVCAircraft(root=base_pth+'aircrafts', split='train', transform=train_T, download=True), datasets.FGVCAircraft(root=base_pth+'aircrafts', split='test', transform=test_T, download=True)
        num_classes = len(train.classes)
    elif dataset_name == 'cub-200': 
        train,test = Cub2011(root=base_pth+'cub-200', transform=train_T, download=True), Cub2011(root=base_pth+'cub-200', train=False, transform=test_T, download=True)
        num_classes = 200
    elif dataset_name == 'ninco': 
        dataset = torchvision.datasets.ImageFolder(base_pth+dataset_name)
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = len(dataset.classes)
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)) 
        train,test = ManualAugs.custom(dataset=train, transforms=train_T), ManualAugs.custom(test, transforms=test_T)
    elif dataset_name == 'oxford-pets':
        train,test = datasets.OxfordIIITPet(root=base_pth+'oxford-pets', split='trainval', transform=train_T, download=True), Cub2011(root=base_pth+'oxford-pets', split='train', transform=train_T, download=True)
        num_classes = 200
    elif dataset_name == 'ham10000':
        ds1, ds2 = base_pth+dataset_name+'/p1', base_pth+dataset_name+'/p2'
        dataset = HAM10000Dataset(base_pth+dataset_name+'/metadata.csv',[ds1,ds2])
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = 7
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)) 
        train,test = ManualAugs.custom(dataset=train, transforms=train_T), ManualAugs.custom(test, transforms=test_T)
    elif dataset_name == 'esc-50':
        mel_transform = transforms.MelSpectrogram(sample_rate=44100, n_fft=2205,hop_length=441)
        ds_pth = base_pth+dataset_name
        dataset = ESC50Dataset(ds_pth+'/metadata.csv', ds_pth+'/audio', transform=mel_transform)
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = 50
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)) 
        train,test = ManualAugs.custom(dataset=train, transforms=train_T), ManualAugs.custom(test, transforms=test_T)
    else: 
        dataset = torchvision.datasets.ImageFolder(base_pth+dataset_name)
        lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
        num_classes = len(dataset.classes)
        train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed)) 
        train,test = ManualAugs.custom(dataset=train, transforms=train_T), ManualAugs.custom(test, transforms=test_T)

    print('\ntrain length: ', len(train), 'test length: ', len(test), '\n')
    return train,test,num_classes