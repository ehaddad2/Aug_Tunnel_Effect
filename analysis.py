"""
Contains code for extracting model embeddings and performing analysis/visualization
"""
from pathlib import Path
import torch
import torch.nn as nn
import random
import math
import torchvision.transforms as Transforms
from PIL import Image, ImageFilter
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from Utils import ManualAugs

SEED = 30
EMBEDDING_PATH_STR = "/home/elias/Deep Learning/Research/OOD/models/IN-100_Test1/embeddings/res18_0.pth"
#seeding:
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Perform analysis on augmented backbone and probes")
    return parser.parse_args()

def visualize_dataset(dataset, dataset_name, n_samples=5, filename="sampled_images.jpg"):
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples*2, 2))
    mean,std = ManualAugs.get_mean_std(dataset_name)
    mean,std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    for i in range(n_samples):
        img, label = dataset[i]
        img = (img * std) + mean
        img = img.permute(1, 2, 0).clamp_(0, 1).numpy()
        axes[i].imshow(img)
        axes[i].set_title(str(label))
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class Analyzer():
    """
    Performs CS analysis and visualization on model embeddings
    """
    def __init__(self, model, epochs, device):
        self.model = model.to(device)
        self.epochs = epochs
        self.device = device
        assert self.device == 'cuda', 'need cuda devices for embedding analysis'

    def __del__(self):
        torch.cuda.empty_cache()

    def extract_forward_layer(self, dataloader: torch.utils.data.DataLoader, device: torch.device, layer, n_samples) -> torch.Tensor:
        layer_outputs = []
        def hook_fn(module, input, output):
            output = output.view(output.size(0), -1)
            layer_outputs.append(output.cpu())

        hook = layer.register_forward_hook(hook_fn) #apply hook
        with torch.no_grad():
            for batchN, (X, y) in enumerate(dataloader):
                if batchN == n_samples: break
                X = X.to(device)
                _ = self.model(X)
                
        hook.remove()
        layer_outputs = torch.cat(layer_outputs)
        return layer_outputs
    
    def extract_embeddings(self, dataloader: torch.utils.data.DataLoader, device: torch.device, layer, n_samples = float('inf'), n_epochs = 0) -> torch.Tensor:
        ep_embeddings = []
        self.model.eval()
        
        for ep in range(self.epochs):
            if ep == n_epochs: break
            print('Extracting At Epoch: ', ep+1)
            penultimate_samples = self.extract_forward_layer(dataloader, device, layer, n_samples)
            ep_embeddings.append(penultimate_samples)

        ep_embeddings = torch.stack(ep_embeddings)
        return ep_embeddings
    
    def worker(self, gpu_idx, embeddings_chunk, block_size, return_dict, generate_sim_mat_pieces):
        curr_device = torch.device(f'cuda:{gpu_idx}')
        print(f"Started processing on {curr_device}")
        min_vals, max_vals, avg_vals = generate_sim_mat_pieces(embeddings_chunk, curr_device, block_size)
        # Move results to CPU and store in the shared dictionary
        return_dict[gpu_idx] = (min_vals.cpu(), max_vals.cpu(), avg_vals.cpu())
        print(f"Finished processing on {curr_device}")
    
    def generate_sim_mat_pieces(self, embeddings: torch.Tensor, device: torch.device, block_size: int):
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

    def process_distributed_embeddings(self, embeddings: torch.Tensor, block_size: int, num_gpus: int):
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
            p = mp.Process(target=self.worker, args=(gpu_idx, chunk_embeddings, block_size, return_dict))
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
    
    def visualize(self, embeddings, save_fig_pth):
        #E, N, D = 100, 10000, 512
        #epoch_embeddings = torch.randn(E, N, D)
        sMin, sMax, sMean = self.process_distributed_embeddings(embeddings,block_size=8000,num_gpus=3)
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
        plt.savefig(save_fig_pth)

    def CS_Analysis(self, model, train_dataloader, epochs, device, embedding_pth, layer_offset=-2):

        #extract at layer offset
        if not Path.exists(embedding_pth):
            print("\nNo embeddings found for analysis, extracting from scratch.")
            layer_name = list(nn.Sequential(*model.children()))[layer_offset] #get penultimate layer name
            epoch_embeddings = self.extract_embeddings(model, train_dataloader, device, layer_name, n_epochs=epochs)
            torch.save(epoch_embeddings, embedding_pth)

        else:
            print("\nEmbeddings found for analysis, using those.")
            epoch_embeddings = torch.load(embedding_pth)


        #visialize & record
        self.visualize(epoch_embeddings, embedding_pth)



def main():

    
    args = parse_args()
    #analyzer = Analyzer()

if __name__ == '__main__':
    main()

