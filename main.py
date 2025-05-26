import torch.multiprocessing.spawn
import wandb
import argparse
import backbone
import probe
import Utils.Models as Models
import Utils.analysis as analysis
import torch
import os
import pandas as pd
import json
from pathlib import Path
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import pandas as pd, re
import hashlib


SEED = 30

class LoadFromJSON(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        json_args = json.load(values)
        for key, value in json_args.items():
            setattr(namespace, key, value)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a backbone or linear probe model with specific settings.")
    parser.add_argument("--args_file", type=open, action=LoadFromJSON, help="Path to a file containing command-line arguments.")
    args = parser.parse_args()

    # Backbone arguments
    parser.add_argument("--backbone_dataset_base_pth", type=str, required=False, default="./data/ID/", help="Base to backbone data")
    parser.add_argument("--backbone_dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--backbone_architecture", type=str, required=True, help="Model architecture.")
    parser.add_argument("--backbone_pth", type=str, required=True, help="Path to the backbone model.")
    parser.add_argument("--backbone_man_aug_setting", nargs="+", required=True, help="Backbone manual aug binary array.")
    parser.add_argument("--backbone_batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--backbone_lr", type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument("--backbone_wd", type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument("--backbone_label_smoothing", type=float, default=0.1, help="Label smoothing for targets.")
    parser.add_argument("--backbone_epochs", type=int, default=512, help="Number of training epochs.")
    parser.add_argument("--backbone_cuda_devices", nargs="+", type=int, default=[0,1], help="CUDA device IDs to use.")
    parser.add_argument("--backbone_t1Max", type=int, default=1, help="Top-1 max test acc for continuing from checkpoint")

    # Probe arguments
    parser.add_argument("--probe_datasets_base_pth", type=str, required=False, default="./data/OOD/", help="Base path to probe data")
    parser.add_argument("--probe_datasets", nargs="+", default=["all"], help="List of OOD datasets to probe, or 'all' for all.")
    parser.add_argument("--probe_pth", type=str, required=True, help="Base path to save the trained linear probe.")
    parser.add_argument("--probe_architecture", type=str, required=True, help="Probing architecture.")
    parser.add_argument("--probe_layers", nargs="+", required=True, help="Layers to probe on. Put 'all' to probe all layers.")
    parser.add_argument("--probe_batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--probe_lr", type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument("--probe_label_smoothing", type=float, default=0.1, help="Label smoothing for targets.")
    parser.add_argument("--probe_epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--probe_cuda_devices", nargs="+", type=int, default=[0], help="CUDA device IDs to use (ONLY USE 0 for now)")
    
    # Shared arguments
    parser.add_argument("--use_wandb", type=bool, default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--run_ID", type=str, default=None, help="Run ID, if empty will be created automatically")
    parser.add_argument("--run_name", type=str, default=None, help="Run name, if empty will be created automatically")
    parser.add_argument("--run_ID_version", type=str, default="0", help="Run ID version (since deleted runs need a new one)")
    parser.add_argument("--use_ddp", type=bool, default=False, help="Train model on multiple GPUs using DDP paradigm")
    parser.add_argument("--use_tpu", type=bool, default=False, help="Set to true if training on TPUs") 
    parser.add_argument("--img_dims", type=int, default=False, help="Cropping dim for images")
    parser.add_argument("--loader_workers", type=int, default=0, help="Number of worker processes for each dataloader")
    return args

def get_probe_dataset_names(args):
    id_ds_name, ood_ds_names = args.backbone_dataset_name, args.preset_ood_datasets if (args.probe_datasets and args.probe_datasets[0]=='all') else args.probe_datasets
    datasets = [id_ds_name] + ood_ds_names
    return datasets

def encode_vector(vector):
    vector_str = ','.join(map(str, vector))
    hash_object = hashlib.sha256(vector_str.encode())
    scalar = int(hash_object.hexdigest(), 16) % 10000
    return scalar

def extract_run_name(backbone_pth):
    path_parts = backbone_pth.split('/')
    model = path_parts[-3]
    dataset = path_parts[-2]
    mode = path_parts[-1].split(':')[-1].replace('.pth', '')
    run_name = f"{model} + {dataset} + {mode}"
    return run_name

if __name__ == '__main__':
    args = parse_args()
    device = None
    if not args.use_tpu: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mp.set_start_method('spawn', force=True)
    print(f'\nDevice being used: ', device if device else 'TPU', '\n')
    
    run_id = f"backbone_{args.backbone_architecture}-{args.backbone_dataset_name}-man_aug_{encode_vector(args.backbone_man_aug_setting)}" if args.run_ID=="" else args.run_ID
    run_name = extract_run_name(args.backbone_pth)
    if args.use_wandb:
        wandb_run_id = run_id + f'-v{args.run_ID_version}'
        wandb.init(
            project="Aug & Tunnel Effect",
            id=wandb_run_id,
            resume='allow',
            name=run_name if (not args.run_name or args.run_name == "") else args.run_name,
            config=vars(args))
    
    visualized_fig = analysis.visualize_dataset(args.backbone_dataset_base_pth, args.backbone_dataset_name, man_aug=args.backbone_man_aug_setting, filename="./figures/sampled_images.jpg")

    """
    -----------------|
    Backbone Training|
    -----------------|
    """
    backbone_results = None

    if not Path.exists(Path(args.backbone_pth)):
        manager = Manager()
        backbone_ret = manager.dict()
        
        if device:
            if (('cpu' in device.type) or ('cuda' in device.type)) and not args.use_ddp: #either cpu or DP training
                backbone_ret = backbone.cpu_worker(
                device,
                args.loader_workers,
                args.backbone_dataset_base_pth,
                args.backbone_dataset_name,
                args.backbone_architecture,
                args.backbone_pth, 
                args.backbone_man_aug_setting,
                args.img_dims,
                args.backbone_lr,
                args.backbone_label_smoothing,
                args.backbone_epochs,
                args.backbone_batch_size,
                cuda_devices=args.backbone_cuda_devices)

            elif ('cuda' in device.type) and args.use_ddp: #DDP for cuda
                mp.spawn(backbone.ddp_worker, args= (
                len(args.backbone_cuda_devices),
                args.loader_workers,
                args.backbone_dataset_base_pth,
                args.backbone_dataset_name,
                args.backbone_architecture,
                args.backbone_pth, 
                args.backbone_man_aug_setting,
                args.img_dims,
                args.backbone_lr,
                args.backbone_label_smoothing,
                args.backbone_epochs,
                args.backbone_batch_size,
                backbone_ret), nprocs=len(args.backbone_cuda_devices))

            else: NotImplementedError(f"Device type '{device.type}' is not supported.")
        
        else: #TPU
            try: 
                import torch_xla.distributed.xla_multiprocessing as xmp
                import torch_xla.core.xla_model as xm
                
                xmp.spawn(backbone.tpu_worker, args=(
                    args.loader_workers,
                    args.backbone_dataset_base_pth,
                    args.backbone_dataset_name,
                    args.backbone_architecture,
                    args.backbone_pth, 
                    args.backbone_man_aug_setting,
                    args.img_dims,
                    args.backbone_lr,
                    args.backbone_label_smoothing,
                    args.backbone_epochs,
                    args.backbone_batch_size,
                    backbone_ret), nprocs=None)
            except ImportError:
                print('ERROR: cannot train backbone with TPUs')
                exit(1)

        backbone_results = backbone_ret[0]
        if args.use_wandb and backbone_results:
            if visualized_fig: 
                wandb.log({"dataset_samples": wandb.Image(visualized_fig)})
            
            for epoch, (train_acc, test_acc) in enumerate(zip(backbone_results['train_acc'], backbone_results['test_acc'])):
                wandb.log({
                    "Backbone Train Accuracy": train_acc,
                    "Backbone Test Accuracy": test_acc,
                    "epoch": epoch + 1
                })
            
            for epoch, (train_loss, test_loss) in enumerate(zip(backbone_results['train_loss'], backbone_results['test_loss'])):
                wandb.log({
                    "Backbone Train Loss": train_loss,
                    "Backbone Test Loss": test_loss,
                    "epoch": epoch + 1
                })

        backbone_acc = backbone_results['max_test_acc']

        # gather summary info & save
        analysis.summarize_backbone_experiments(wandb, run_name, args.backbone_architecture, args.backbone_man_aug_setting, backbone_acc)
    else: 
        print(f"Backbone {args.backbone_pth} found, probing with this.")
        backbone_acc = args.backbone_t1Max

    """
    -----------------|
    Probe Training   |
    -----------------|
    """
    probe_results = {}  # {dataset: [layer1_acc, layer2_acc, ...]}
    probing_datasets = get_probe_dataset_names(args)
    probe_layers = Models.get_all_probe_layer_names(args) if (args.probe_layers and str.lower(args.probe_layers[0]) == 'all') else args.probe_layers
    manager = Manager()
    for i in range(len(probing_datasets)):
        probe_results[probing_datasets[i]] = []
        for j in range(len(probe_layers)):
            full_probe_pth = args.probe_pth + "/" + args.backbone_architecture + "/" + args.backbone_dataset_name + "/" + "man_aug:" + str(encode_vector(args.backbone_man_aug_setting)) + "/" +  probing_datasets[i] + "/" + str(args.probe_architecture) + "/" + probe_layers[j] if probe_layers else probe_layers
            print(f'\nProbing dataset: {probing_datasets[i]} at probe layer: {probe_layers[j]}')
            probe_ret = None 
            if device:
                if (('cpu' in device.type) or ('cuda' in device.type)) and not args.use_ddp:
                    probe_ret = probe.cpu_worker(
                        device,
                        args.loader_workers,
                        args.probe_datasets_base_pth if i>0 else args.backbone_dataset_base_pth,
                        probing_datasets[i],
                        args.backbone_dataset_name,
                        args.backbone_pth,
                        args.backbone_architecture,
                        full_probe_pth,
                        args.probe_architecture,
                        probe_layers[j] if probe_layers else probe_layers,
                        args.img_dims,
                        args.probe_lr,
                        args.probe_label_smoothing,
                        args.probe_epochs,
                        args.probe_batch_size,
                        cuda_devices=args.probe_cuda_devices)

                elif ('cuda' in device.type) and args.use_ddp:
                    probe_ret = manager.dict()
                    mp.spawn(probe.ddp_worker, args=(
                        len(args.probe_cuda_devices),
                        args.loader_workers,
                        args.probe_datasets_base_pth if i>0 else args.backbone_dataset_base_pth,
                        probing_datasets[i],
                        args.backbone_dataset_name,
                        args.backbone_pth,
                        args.backbone_architecture,
                        full_probe_pth,
                        args.probe_architecture,
                        probe_layers[j] if probe_layers else probe_layers,
                        args.img_dims,
                        args.probe_lr,
                        args.probe_label_smoothing,
                        args.probe_epochs,
                        args.probe_batch_size,
                        probe_ret
                    ), nprocs=len(args.probe_cuda_devices))

                else: raise NotImplementedError(f"Device type '{device.type}' is not supported.")

            else:
                try:
                    import torch_xla.distributed.xla_multiprocessing as xmp
                    import torch_xla.core.xla_model as xm
                    probe_ret = manager.dict()
                    xmp.spawn(probe.tpu_worker, args=(
                        args.loader_workers,
                        args.probe_datasets_base_pth if i>0 else args.backbone_dataset_base_pth,
                        probing_datasets[i],
                        args.backbone_dataset_name,
                        args.backbone_pth,
                        args.backbone_architecture,
                        full_probe_pth,
                        args.probe_architecture,
                        probe_layers[j] if probe_layers else probe_layers,
                        args.img_dims,
                        args.probe_lr,
                        args.probe_label_smoothing,
                        args.probe_epochs,
                        args.probe_batch_size,
                        probe_ret), nprocs=None)
                except ImportError:
                    print('ERROR: cannot probe with TPUs')
                    exit(1)
                
            #collect results after probing one layer
            if probe_ret:
                probe_ret = probe_ret[0]
                probe_results[probing_datasets[i]].append(probe_ret['max_test_acc'])
            else: 
                print(f"No probing results for dataset {probing_datasets[i]} at layer {probe_layers[j]}")
            

    print(f'\nProbed all datasets for backbone.')

    """
    -----------------|
    Analysis         |
    -----------------|
    """

    """
                    df = pd.DataFrame(results, columns=[
                    "Test_Num",
                    "Layer_Num",
                    "Backbone Architecture",
                    "Manual Augmentation Setting",
                    "ID Dataset",
                    "OOD Dataset",
                    "Backbone ID max top-1 test acc",
                    "Probe max top-1 test acc"
                ])  
                wandb.log({"Run Results": wandb.Table(dataframe=df)})
    """
    print(f'Probe Results Dict: {probe_results}')
    
    if probe_results:
        # gather summary info for probes & save
        ood_accs_list = []
        probe_model = re.search(r'\d+', args.backbone_dataset_name)
        id_class_count = int(probe_model.group())
        id_layer_res = probe_results[args.backbone_dataset_name]
        id_ds = probing_datasets[0]
        for ood_ds in probing_datasets: #for each OOD dataset, we need ID acc and OOD acc vectors (for that dataset) to find 3 metrics, and plot them all on this row
            if ood_ds == args.backbone_dataset_name: continue
            ood_layer_res = probe_results[ood_ds]
            print(f'ID layer res: {id_layer_res}\nOOD layer res: {ood_layer_res}')
            r, rho, A = analysis.compute_OOD_metrics(id_layer_res, ood_layer_res, id_ds, ood_ds, id_class_count)
            analysis.summarize_probe_experiments(wandb, extract_run_name(args.backbone_pth), ood_ds, args.backbone_man_aug_setting, r, rho, A)
            
    if args.use_wandb: wandb.finish()
    