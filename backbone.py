import os
import sys
sys.path.append('/home/elias/Deep Learning/Utils')
import math
import torch
import torch.utils.data
from torchinfo import summary
import random
from torchmetrics import Accuracy
import torch.utils
from torch.utils.data import DataLoader, Subset, Dataset
import DatasetPrep
import torchvision
import torchvision.transforms as Transforms
from PIL import Image
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
from timeit import default_timer as timer
from pathlib import Path
import TrainModel
import matplotlib.pyplot as plt
import tqdm
import torch.multiprocessing as mp

SEED = 30
BATCH_SIZE = 512
EPOCHS = 100
ID_DATASET_PATH = "/home/elias/Deep Learning/Research/OOD/data/imagenet-100"
MODEL_PATH_STR = "/home/elias/Deep Learning/Research/OOD/models/IN-100_Test1/res18_3.pth"
EMBEDDING_PATH_STR = "/home/elias/Deep Learning/Research/OOD/models/IN-100_Test1/embeddings/res18_0.pth"
MODEL_SKELETON = 'resnet18'
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
    
class ResNet10(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet10, self).__init__(
            block=BasicBlock, #type of block to make
            layers=[1, 1, 1, 1],  #number of blocks per layer
            num_classes=num_classes)

def generate_sim_mat_pieces(embeddings: torch.Tensor, device: torch.device, block_size: int):
    E, N = embeddings.shape[0], embeddings.shape[1]
    embeddings = embeddings.view(E * N, -1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-12).to(device)
    num_blocks = (E * N) // block_size
    
    min_vals = torch.full((E * N,), float('inf')).to(device)
    max_vals = torch.full((E * N,), float('-inf')).to(device)
    summed_vals = torch.zeros((E * N,), device=device)
    count_vals = torch.zeros((E * N,), device=device)
    
    for i in tqdm.tqdm(range(num_blocks)):
        for j in range(num_blocks):
            slice1 = embeddings[i * block_size:(i + 1) * block_size]
            slice2 = embeddings[j * block_size:(j + 1) * block_size]
            sim = torch.matmul(slice1, slice2.t())
            extracted_min = torch.min(sim, dim=1)[0]
            if i == j: sim.fill_diagonal_(0) 
            extracted_max = torch.max(sim, dim=1)[0]
            extracted_sum = torch.sum(sim, dim=1)
            block_count = sim.size(1)
            
            min_vals[i * block_size:(i + 1) * block_size] = torch.min(min_vals[i * block_size:(i + 1) * block_size], extracted_min)
            max_vals[i * block_size:(i + 1) * block_size] = torch.max(max_vals[i * block_size:(i + 1) * block_size], extracted_max)
            summed_vals[i * block_size:(i + 1) * block_size] += extracted_sum
            count_vals[i * block_size:(i + 1) * block_size] += block_count

    avg_vals = summed_vals / count_vals
    return min_vals, max_vals, avg_vals
    
def worker(gpu_idx, embeddings_chunk, block_size, return_dict):
    curr_device = torch.device(f'cuda:{gpu_idx}')
    print(f"Started processing on {curr_device}")
    min_vals, max_vals, avg_vals = generate_sim_mat_pieces(embeddings_chunk, curr_device, block_size)
    # Move results to CPU and store in the shared dictionary
    return_dict[gpu_idx] = (min_vals.cpu(), max_vals.cpu(), avg_vals.cpu())
    print(f"Finished processing on {curr_device}")
def process_distributed_embeddings2(embeddings: torch.Tensor, block_size: int, num_gpus: int):
    """
    Split embeddings across available GPUs, process each chunk in parallel, then combine results.
    """
    E, N = embeddings.shape[0], embeddings.shape[1]
    chunk_size = int(math.ceil(E / num_gpus))
    all_mins, all_maxs, all_means = [], [], []
    torch.cuda.empty_cache()
    
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Create a manager to store results from all processes
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for gpu_idx in range(num_gpus):
        start_idx = gpu_idx * chunk_size
        end_idx = min(start_idx + chunk_size, E)
        if start_idx >= E: break
        chunk_embeddings = embeddings[start_idx:end_idx]
        
        # Start a new process for each GPU
        p = mp.Process(target=worker, args=(gpu_idx, chunk_embeddings, block_size, return_dict))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Collect results from all processes
    for gpu_idx in sorted(return_dict.keys()):
        min_vals, max_vals, avg_vals = return_dict[gpu_idx]
        all_mins.append(min_vals)
        all_maxs.append(max_vals)
        all_means.append(avg_vals)
    
    final_mins = torch.cat(all_mins)
    final_maxs = torch.cat(all_maxs)
    final_means = torch.cat(all_means)
    return final_mins, final_maxs, final_means

def main():
        
    """
    Functions
    """
    def collate_fn(data):
        images, labels = zip(*data)
        max_height = max([img.size(1) for img in images])
        max_width = max([img.size(2) for img in images])
        padded_images = torch.zeros((len(images), 3, max_height, max_width))

        # Pad each image
        for i, img in enumerate(images):
            c, h, w = img.size()
            padded_images[i, :, :h, :w] = img #place in top left

        labels = torch.tensor(labels)
        return padded_images.float(), labels.long()
    
    def print_train_time(start: float, end: float, device: torch.device = None):
        total_time = end - start
        print(f"Train time on {device}: {total_time:.3f} seconds")
        return total_time

    def extract_forward_layer(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, layer, n_samples) -> torch.Tensor:
        layer_outputs = []
        def hook_fn(module, input, output):
            output = output.view(output.size(0), -1)
            layer_outputs.append(output.cpu())

        hook = layer.register_forward_hook(hook_fn) #apply hook
        with torch.no_grad():
            for batchN, (X, y) in enumerate(dataloader):
                if batchN == n_samples: break
                X = X.to(device)
                _ = model(X)
                
        hook.remove()
        layer_outputs = torch.cat(layer_outputs)
        return layer_outputs

    def extract_embeddings(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, layer, n_samples = float('inf'), n_epochs = 0) -> torch.Tensor:
        ep_embeddings = []
        model = nn.DataParallel(model, [0])
        model.to(device)
        model.eval()
        
        for ep in range(EPOCHS):
            if ep == n_epochs: break
            print('Extracting At Epoch: ', ep+1)
            penultimate_samples = extract_forward_layer(model, dataloader, device, layer, n_samples)
            ep_embeddings.append(penultimate_samples)

        ep_embeddings = torch.stack(ep_embeddings)
        return ep_embeddings

    def generate_sim_mat(embeddings: torch.Tensor, device: torch.device):
        E,N = embeddings.shape[0], embeddings.shape[1]
        embeddings = embeddings.view(E*N,-1).to(device)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1, eps=1e-12).to(device)
        sim = torch.matmul(embeddings, embeddings.t()).to(device)
        return sim



    def process_distributed_embeddings(embeddings: torch.Tensor, block_size: int, num_gpus: int):
        """
        Split embeddings across available GPUs, process each chunk, then combine results.
        """
        E, N = embeddings.shape[0], embeddings.shape[1]
        chunk_size = E // num_gpus 
        all_mins, all_maxs, all_means = [], [], []
        torch.cuda.empty_cache()
        for gpu_idx in range(num_gpus):
            start_idx = gpu_idx * chunk_size
            end_idx = start_idx + chunk_size if gpu_idx < num_gpus - 1 else E
            
            # process chunks on gpu's
            
            curr_device = torch.device(f'cuda:{gpu_idx}')
            print(f"started on {curr_device}")
            chunk_embeddings = embeddings[start_idx:end_idx]
            min_vals, max_vals, avg_vals = generate_sim_mat_pieces(chunk_embeddings, curr_device, block_size)
            
            # move res to cpu
            all_mins.append(min_vals.cpu())
            all_maxs.append(max_vals.cpu())
            all_means.append(avg_vals.cpu())
            del chunk_embeddings, min_vals, max_vals, avg_vals
            torch.cuda.empty_cache()
            print(f"ended on {curr_device}")
        
        final_mins = torch.cat(all_mins)
        final_maxs = torch.cat(all_maxs)
        final_means = torch.cat(all_means)
        return final_mins, final_maxs, final_means
    

    """
    Pretrained Models
    """
    weights = torchvision.models.ResNet18_Weights.DEFAULT

    """
    Data
    """
    transforms0 = Transforms.Compose([
        Transforms.Resize(256, interpolation=Transforms.InterpolationMode.BILINEAR),
        Transforms.CenterCrop(224),
        Transforms.ToTensor(),
        Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    ])
    transforms1 = Transforms.Compose([
        Transforms.RandomHorizontalFlip(p=1)
    ])
    transforms2 = Transforms.Compose([
        Transforms.RandomResizedCrop((224,224))
    ])
    transforms3 = Transforms.Compose([
        Transforms.RandomHorizontalFlip(p=1),
        Transforms.RandomResizedCrop((224,224))
    ])
    transforms4 = Transforms.Compose([
        Transforms.RandomGrayscale(p=1)
    ])

    dataset = torchvision.datasets.ImageFolder(ID_DATASET_PATH)
    lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
    train, test = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(SEED)) 
    N_CLASSES = len(dataset.classes)
    train = DatasetClone(train, transform=transforms0)
    test = DatasetClone(test, transform=transforms0)


    print('train length: ', len(train), 'test length: ', len(test), '\n')

    train_dataloader = DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=os.cpu_count())
    test_dataloader = DataLoader(test, BATCH_SIZE, shuffle=False, num_workers=os.cpu_count())
    img, label = next(iter(train_dataloader))
    print(f'\nBatch dimensions:', img.shape, '\n')

    """
    Models
    """
    if str.lower(MODEL_SKELETON) == "resnet10":
        model = ResNet10()
    else:
        model = torchvision.models.resnet18()

    model.fc = nn.Linear(512, N_CLASSES)

    """


    pretrained_model = torchvision.models.resnet18(weights=weights)
    pretrained_model.fc = nn.Linear(512, numClasses)
    pretrained_penultimate = list(nn.Sequential(*pretrained_model.children()))[-2] #get penultimate layer name
    """
    penultimate = list(nn.Sequential(*model.children()))[-2] #get penultimate layer name

    """
   
    """
    model.load_state_dict(torch.load(MODEL_PATH_STR))
    model.to(device)
    #summary(model, input_size=img.shape, col_names=["trainable", "input_size", "output_size", "kernel_size"])


    """
    Train/Test
    """
    img, label = next(iter(train_dataloader))
    _,C,H,W = np.array(img.shape)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
    accuracy_fn = Accuracy(task='multiclass', num_classes=N_CLASSES)

    trainStartTime = timer()
    #print(TrainModel.test_step(model, test_dataloader, loss_fn, device))

    #results = TrainModel.train(model, train_dataloader, test_dataloader, opt, loss_fn, EPOCHS, device, None, 5, True, cuda_devices=[0,1,2])

    #torch.save(model.state_dict(), MODEL_PATH_STR)

    #epoch_embeddings = extract_embeddings(model, train_dataloader, device, penultimate, n_epochs=EPOCHS)
    #torch.save(epoch_embeddings, EMBEDDING_PATH_STR)
    

    """
    Visualization
    """
    assert device == 'cuda', 'need cuda devices for embedding analysis'
    #E, N, D = 100, 10000, 512
    #epoch_embeddings = torch.randn(E, N, D)
    epoch_embeddings = torch.load(EMBEDDING_PATH_STR)
    sMin, sMax, sMean = process_distributed_embeddings2(
        epoch_embeddings,
        block_size=8000,
        num_gpus=3
    )
    unique_samples = 0
    sMin,sMax,sMean = sMin.cpu(),sMax.cpu(),sMean.cpu()
    unique_samples = (sMax <= 0.9).sum().item()
    print("Number of unique samples: ", unique_samples, '\n')

    downsample_factor = 30
    sMin_downsampled = sMin[::downsample_factor]
    sMax_downsampled = sMax[::downsample_factor]
    sMean_downsampled = sMean[::downsample_factor]

    # Plotting the smoothed line charts
    plt.plot(sMin_downsampled, color='blue', label='Min')
    plt.plot(sMax_downsampled, color='green', label='Max')
    plt.plot(sMean_downsampled, color='red', label='Mean')

    plt.xlabel(f'Samples (unique samples = {unique_samples})', fontweight='bold')
    plt.ylabel('CS', fontweight='bold')
    plt.title('ResNet18 on pre-trained IN1K (no-aug)', fontweight='bold')

    plt.legend()
    plt.savefig("/home/elias/Deep Learning/Research/OOD/figures/IN1k_0.png")

    exit()

    trainEndTime = timer()
    totalTrainTime = print_train_time(trainStartTime, trainEndTime, str(next(model.parameters()).device))
    writer.close()
    explainer = shap.Explainer(model)
    shapVals = explainer(train_data)
    shap.bar_plot(shapVals)
    #visualize
    evalPic = transformations(Image.open(Path("data/testStk.jpeg"))).unsqueeze(0)

    print(evalPic.shape)
    print(model(evalPic))


if __name__ == '__main__':
    main()
    