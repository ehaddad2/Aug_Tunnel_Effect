import torch.multiprocessing.spawn
import wandb
import argparse
import backbone
import probe
import Models
import analysis
import torch
import os
import pandas as pd
import json
from pathlib import Path
import analysis
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
    parser.add_argument("--backbone_manual_aug_setting", nargs="+", required=True, help="Backbone manual aug binary array.")
    parser.add_argument("--backbone_aug_policy_setting", nargs="+", required=True, help="Backbone aug policy binary array.")
    parser.add_argument("--backbone_batch_size", type=int, default=512, help="Batch size for training.")
    parser.add_argument("--backbone_lr", type=float, default=0.01, help="Learning rate for optimizer.")
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
    parser.add_argument("--run_name", type=str, default="Untitled Run", help="Name W&B run.")
    parser.add_argument("--run_ID", type=str, default=None, help="Run ID, if empty will be created automatically")
    parser.add_argument("--run_ID_version", type=str, default="0", help="Run ID version (since deleted runs need a new one)")
    parser.add_argument("--use_ddp", type=bool, default=False, help="Train model on multiple GPUs using DDP paradigm")
    parser.add_argument("--use_tpu", type=bool, default=False, help="Set to true if training on TPUs") 
    parser.add_argument("--img_dims", type=int, default=False, help="Cropping dim for images")
    parser.add_argument("--loader_workers", type=int, default=0, help="Number of worker processes for each dataloader")
    return args

def get_probe_dataset_names(base_pths, id_ds_name):
    datasets = [id_ds_name, 'aircrafts', 'cifar-10', 'cub-200', 'flowers-102', 'stl-10', 'ninco', 'ham10000', 'esc-50']
    ret = []
    for base_pth in base_pths: #add in current datasets
        if not os.path.isdir(base_pth): raise NotADirectoryError(f"{base_pth} is not a directory.")
        else: ret += [filename for filename in os.listdir(base_pth)]
    
    for ds in datasets: #add in missing ones
        if ds not in ret: ret.append(ds)

    return ret

def encode_vector(vector):
    vector_str = ','.join(map(str, vector))
    hash_object = hashlib.sha256(vector_str.encode())
    scalar = int(hash_object.hexdigest(), 16) % 10000  # 4-digit scalar
    return scalar

if __name__ == '__main__':
    args = parse_args()
    device = None
    if not args.use_tpu: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mp.set_start_method('spawn', force=True)
    print(f'\nDevice being used: ', device if device else 'TPU', '\n')
    
    run_id = f"backbone_{args.backbone_architecture}-{args.backbone_dataset_name}-man_aug_{encode_vector(args.backbone_man_aug_setting)}-policy_aug_{encode_vector(args.backbone_aug_policy_setting)}" if args.run_ID=="" else args.run_ID
    if args.use_wandb:
        wandb_run_id = run_id + f'-v{args.run_ID_version}'
        wandb.init(
            project="Aug & Tunnel Effect",
            id=wandb_run_id,
            resume='allow',
            name=args.run_name,
            config=vars(args))
    
    visualized_fig = analysis.visualize_dataset(args.backbone_dataset_base_pth, args.backbone_dataset_name, man_aug=args.backbone_man_aug_setting, aug_policy=args.backbone_aug_policy_setting, filename="./figures/sampled_images.jpg")

    """
    -----------------|
    Backbone Training|
    -----------------|
    """
    backbone_results = None
    probe_layers = Models.get_all_probe_layer_names(args.backbone_architecture) if (args.probe_layers and str.lower(args.probe_layers[0]) == 'all') else args.probe_layers
    if not Path.exists(Path(args.backbone_pth)) or 'test' in args.backbone_pth:
        manager = Manager()
        backbone_ret = manager.dict()
        
        if device:
            if (('cpu' in device.type) or ('cuda' in device.type)) and not args.ddp: #either cpu or DP training
                backbone_ret = backbone.cpu_worker(
                device,
                args.loader_workers,
                args.backbone_dataset_base_pth,
                args.backbone_dataset_name,
                args.backbone_architecture,
                args.backbone_pth, 
                args.backbone_man_aug_setting,
                args.backbone_aug_policy_setting,
                args.img_dims,
                args.backbone_lr,
                args.backbone_label_smoothing,
                args.backbone_epochs,
                args.backbone_batch_size,
                cuda_devices=args.backbone_cuda_devices)

            elif ('cuda' in device.type) and args.ddp: #DDP for cuda
                mp.spawn(backbone.ddp_worker, args= (
                args.loader_workers,
                args.backbone_dataset_base_pth,
                args.backbone_dataset_name,
                args.backbone_architecture,
                args.backbone_pth, 
                args.backbone_man_aug_setting,
                args.backbone_aug_policy_setting,
                args.img_dims,
                args.backbone_lr,
                args.backbone_label_smoothing,
                args.backbone_epochs,
                args.backbone_batch_size,
                backbone_ret), nprocs=len(args.backbone_cuda_devices))

            else: NotImplementedError(f"Device type '{device.type}' is not supported.")
        
        else: #use TPU
            import torch_xla.distributed.xla_multiprocessing as xmp
            import torch_xla.core.xla_model as xm
            xmp.spawn(backbone.tpu_worker, args=(
                args.loader_workers,
                args.backbone_dataset_base_pth,
                args.backbone_dataset_name,
                args.backbone_architecture,
                args.backbone_pth, 
                args.backbone_man_aug_setting,
                args.backbone_aug_policy_setting,
                args.img_dims,
                args.backbone_lr,
                args.backbone_label_smoothing,
                args.backbone_epochs,
                args.backbone_batch_size,
                backbone_ret), nprocs=None)

        backbone_results = backbone_ret[0]
        
        if args.use_wandb and backbone_results:
            if visualized_fig:
                wandb.log({"dataset_samples": wandb.Image(visualized_fig)})
            backbone_accuracy_data = [
                [epoch + 1, value, series]
                for epoch, (train_acc, test_acc) in enumerate(zip(backbone_results['train_acc'], backbone_results['test_acc']))
                for value, series in zip([train_acc, test_acc], ["Backbone Train Accuracy", "Backbone Test Accuracy"])
            ]
            backbone_loss_data = [
                [epoch + 1, value, series]
                for epoch, (train_loss, test_loss) in enumerate(zip(backbone_results['train_loss'], backbone_results['test_loss']))
                for value, series in zip([train_loss, test_loss], ["Backbone Train Loss", "Backbone Test Loss"])
            ]

            backbone_accuracy_chart = wandb.plot.line(
                table=wandb.Table(data=backbone_accuracy_data, columns=["Epoch", "Value", "Series"]),
                x="Epoch",
                y="Value",
                stroke="Series",
                title="Backbone Accuracy Over Epochs"
            )
            backbone_loss_chart = wandb.plot.line(
                table=wandb.Table(data=backbone_loss_data, columns=["Epoch", "Value", "Series"]),
                x="Epoch",
                y="Value",
                stroke="Series",
                title="Backbone Loss Over Epochs"
            )

            # Log only the charts
            wandb.log({
                "Backbone Accuracy Chart": backbone_accuracy_chart,
                "Backbone Loss Chart": backbone_loss_chart
            })
        backbone_acc = backbone_results['max_test_acc']
        # gather summary info & save
        backbone_csv_dir = Path("./csv_results/")
        backbone_csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = backbone_csv_dir / 'Backbones.csv'
        m = re.search(r'\d+', args.backbone_dataset_name)
        id_class_count = int(m.group()) if m else None
        overparam_lvl = analysis.compute_overparam_val(args.backbone_architecture, args.backbone_dataset_base_pth, args.backbone_dataset_name)

        analysis.summarize_backbone_experiments(run_id, csv_path, args.backbone_architecture, args.backbone_man_aug_setting, args.backbone_aug_policy_setting,
                                                args.img_dims, id_class_count, overparam_lvl, len(probe_layers), backbone_acc)
    else: 
        print(f"Backbone {args.backbone_pth} found, probing with this.")
        backbone_acc = args.backbone_t1Max

    """
    -----------------|
    Probe Training   |
    -----------------|
    """
    probe_results = {}  # {dataset: [layer1_acc, layer2_acc, ...]}
    probing_datasets = get_probe_dataset_names([args.backbone_dataset_base_pth, args.probe_datasets_base_pth], args.backbone_dataset_name) if (args.probe_datasets and str.lower(args.probe_datasets[0]) == 'all') else args.probe_datasets
    manager = Manager()
    print(f'Probing: {[ds for ds in probing_datasets]}')
    for i in range(len(probing_datasets)):
        probe_results[probing_datasets[i]] = []
        for j in range(len(probe_layers)):
            full_probe_pth = args.probe_pth + "/" + args.backbone_architecture + "/" + args.backbone_dataset_name + "/" + "man_aug:" + str(args.backbone_man_aug_setting)  + " aug_policy:" + str(args.backbone_aug_policy_setting) + "/" +  probing_datasets[i] + "/" + str(args.probe_architecture) + "/" + probe_layers[j]
            #if Path.exists(Path(full_probe_pth)) and not (probing_datasets[i] == args.backbone_dataset_name):
                #print(f'\nProbed dataset: {probing_datasets[i]}, moving to next...')
                #continue
            print(probe_layers)
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
                        probe_layers[j],
                        args.img_dims,
                        args.probe_lr,
                        args.probe_label_smoothing,
                        args.probe_epochs,
                        args.probe_batch_size,
                        cuda_devices=args.probe_cuda_devices)

                elif ('cuda' in device.type) and args.use_ddp:
                    probe_ret = manager.dict()
                    mp.spawn(probe.ddp_worker, args=(
                        args.loader_workers,
                        args.probe_datasets_base_pth if i>0 else args.backbone_dataset_base_pth,
                        probing_datasets[i],
                        args.backbone_dataset_name,
                        args.backbone_pth,
                        args.backbone_architecture,
                        full_probe_pth,
                        args.probe_architecture,
                        probe_layers[j],
                        args.img_dims,
                        args.probe_lr,
                        args.probe_label_smoothing,
                        args.probe_epochs,
                        args.probe_batch_size,
                        probe_ret
                    ), nprocs=len(args.probe_cuda_devices))

                else: raise NotImplementedError(f"Device type '{device.type}' is not supported.")

            else:
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
                    probe_layers[j],
                    args.img_dims,
                    args.probe_lr,
                    args.probe_label_smoothing,
                    args.probe_epochs,
                    args.probe_batch_size,
                    probe_ret), nprocs=4)

            if probe_ret: probe_ret = probe_ret[0]
            if args.use_wandb and probe_ret:
                accuracy_data = [
                    [epoch + 1, value, series]
                    for epoch, (train_acc, test_acc) in enumerate(
                        zip(probe_ret['train_acc'], probe_ret['test_acc'])
                    )
                    for value, series in zip([train_acc, test_acc], ["Train Accuracy", "Test Accuracy"])
                ]
                loss_data = [
                    [epoch + 1, value, series]
                    for epoch, (train_loss, test_loss) in enumerate(
                        zip(probe_ret['train_loss'], probe_ret['test_loss'])
                    )
                    for value, series in zip([train_loss, test_loss], ["Train Loss", "Test Loss"])
                ]
                accuracy_table = wandb.Table(data=accuracy_data, columns=["Epoch", "Value", "Series"])
                loss_table = wandb.Table(data=loss_data, columns=["Epoch", "Value", "Series"])

                accuracy_chart = wandb.plot.line(
                    table=accuracy_table,
                    x="Epoch",
                    y="Value",
                    stroke="Series",  # Group lines by "Series" (Train Accuracy, Test Accuracy)
                    title=f"{probing_datasets[i]} Probe Accuracy Over Epochs"
                )

                loss_chart = wandb.plot.line(
                    table=loss_table,
                    x="Epoch",
                    y="Value",
                    stroke="Series",  # Group lines by "Series" (Train Loss, Test Loss)
                    title=f"{probing_datasets[i]} Probe Loss Over Epochs"
                )
                wandb.log({
                    f"{probing_datasets[i]} Accuracy Chart": accuracy_chart,
                    f"{probing_datasets[i]} Loss Chart": loss_chart
                })

                #prep final results
                results = []
                results.append([
                    i+1,
                    j+1,
                    args.backbone_architecture,
                    str(args.backbone_man_aug_setting),
                    str(args.backbone_aug_policy_setting),
                    args.backbone_dataset_name,
                    probing_datasets[i],
                    backbone_acc,
                    probe_ret['max_test_acc'],
                    len(probe_layers) #TODO: FIX
                ])

                df = pd.DataFrame(results, columns=[
                    "Test_Num",
                    "Layer_Num",
                    "Backbone Architecture",
                    "Manual Augmentation Setting",
                    "Augmentation Policy Setting",
                    "ID Dataset",
                    "OOD Dataset",
                    "Backbone ID max top-1 test acc",
                    "Probe max top-1 test acc",
                    "Depth"
                ])  
                wandb.log({"Run Results": wandb.Table(dataframe=df)})

            if probe_ret: probe_results[probing_datasets[i]].append(probe_ret['max_test_acc'])
            
            if not probe_ret: print(f"No probing results for dataset {probing_datasets[i]} at layer {probe_layers[j]}")

    print(f'\nProbed all datasets.')

    """
    -----------------|
    Analysis         |
    -----------------|
    """
    print(f'Probe Results Dict: {probe_results}')
    
    if probe_results:
        # gather summary info for probes & save
        probe_csv_dir = Path("./csv_results")
        if not Path.exists(probe_csv_dir): probe_csv_dir.mkdir(parents=True, exist_ok=True)
        probe_csv_path = Path("./csv_results/Probes.csv")
        ood_accs_list = []
        id_layer_res = probe_results[args.backbone_dataset_name]
        id_ds = probing_datasets[0]
        for ood_ds in probing_datasets: #for each OOD dataset, we need ID acc and OOD acc vectors (for that dataset) to find 3 metrics, and plot them all on this row
            if ood_ds == args.backbone_dataset_name: continue
            ood_layer_res = probe_results[ood_ds]
            print(f'ID layer res: {id_layer_res}\nOOD layer res: {ood_layer_res}')
            r, rho, A = analysis.compute_OOD_metrics(id_layer_res, ood_layer_res, id_ds, ood_ds, id_class_count)
        
            analysis.summarize_probe_experiments(run_id, probe_csv_path, args.backbone_architecture, args.backbone_man_aug_setting, 
                                            args.backbone_aug_policy_setting, args.img_dims, id_class_count, overparam_lvl, len(probe_layers),
                                            backbone_acc, args.probe_architecture, r, rho, A)
    