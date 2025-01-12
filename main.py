import wandb
import argparse
from backbone import BackboneTrainer
from probe import LinearProbeTrainer
import torch
import os
import pandas as pd
import json
from pathlib import Path
import analysis
import torch.multiprocessing as mp
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
    parser.add_argument("--probe_layer", type=str, required=True, help="Layer to probe on.")
    parser.add_argument("--probe_batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--probe_lr", type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument("--probe_label_smoothing", type=float, default=0.1, help="Label smoothing for targets.")
    parser.add_argument("--probe_epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--probe_cuda_devices", nargs="+", type=int, default=[0], help="CUDA device IDs to use (ONLY USE 0 for now)")
    
    # Shared arguments
    parser.add_argument("--use_wandb", type=bool, default=False, help="Enable Weights & Biases logging.")
    parser.add_argument("--run_group", type=str, default="Untitled Run Group", help="Weights & Biases experiment run log group")
    parser.add_argument("--use_ddp", type=bool, default=False, help="Train model on multiple GPUs")
    parser.add_argument("--img_dims", type=int, default=False, help="Cropping dim for images")
    parser.add_argument("--loader_workers", type=int, default=0, help="Number of worker processes for each dataloader")
    parser.add_argument("--dataset_img_pth", type=str, help="Path for sampled backbone dataset images")
    return args

def get_all_dataset_names(base_pth):
    if not os.path.isdir(base_pth): raise NotADirectoryError(f"{base_pth} is not a directory.")
    return [filename for filename in os.listdir(base_pth)]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'\nDevice being used: ', device, '\n')
    args = parse_args()
    
    title = (
    f"backbone_{args.backbone_architecture}+{args.backbone_dataset_name}+man_aug{args.backbone_man_aug_setting}+policy_aug{args.backbone_aug_policy_setting}"
    f"+probe_{args.probe_architecture}")
    results = []
    if args.use_wandb:
        wandb.init(
            project="Aug & Tunnel Effect",
            group=args.run_group,
            name=title,
            config=vars(args)
        )
    
    # Backbone training
    backbone_results = None
    if not Path.exists(Path(args.backbone_pth)):
        trainer = BackboneTrainer(
            dataset_base_pth=args.backbone_dataset_base_pth,
            dataset_name=args.backbone_dataset_name,
            num_workers=(args.loader_workers+len(args.backbone_cuda_devices)-1)//len(args.backbone_cuda_devices),
            architecture=args.backbone_architecture,
            backbone_pth=args.backbone_pth,
            man_aug_setting=args.backbone_man_aug_setting,
            policy_aug_setting=args.backbone_aug_policy_setting,
            img_dims=args.img_dims,
            lr=args.backbone_lr,
            label_smoothing=args.backbone_label_smoothing,
            epochs=args.backbone_epochs,
            cuda_devices=args.backbone_cuda_devices,
            wandb=wandb,
            use_wandb=args.use_wandb,
            device=device,
            seed=SEED)
        
        analysis.visualize_dataset(trainer.train, args.backbone_dataset_name, filename=args.dataset_img_pth)
        if args.use_ddp:
            world_size=len(args.backbone_cuda_devices)
            backbone_results = trainer.main_ddp(world_size, args.backbone_batch_size//world_size)
        else:
            backbone_results = trainer.train_backbone_serial()
        
        if args.use_wandb and backbone_results:
            # Prepare data for charts
            backbone_accuracy_data = [
                [epoch + 1, value, series]
                for epoch, (train_acc, test_acc) in enumerate(
                    zip(backbone_results['train_acc'], backbone_results['test_acc'])
                )
                for value, series in zip([train_acc, test_acc], ["Backbone Train Accuracy", "Backbone Test Accuracy"])
            ]
            backbone_loss_data = [
                [epoch + 1, value, series]
                for epoch, (train_loss, test_loss) in enumerate(
                    zip(backbone_results['train_loss'], backbone_results['test_loss'])
                )
                for value, series in zip([train_loss, test_loss], ["Backbone Train Loss", "Backbone Test Loss"])
            ]

            # Create charts without logging tables
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

    else: 
        print(f"Backbone {args.backbone_pth} found, probing with this.")
        backbone_acc = args.backbone_t1Max

    # Linear probe training
    datasets = get_all_dataset_names(args.probe_datasets_base_pth) if str.lower(args.probe_datasets[0]) == 'all' else args.probe_datasets
    for i in range(len(datasets)):
        
        full_probe_pth = args.probe_pth + "/" + args.backbone_architecture + "/" + args.backbone_dataset_name + "/" + "man_aug:" + str(args.backbone_man_aug_setting)  + " aug_policy:" + str(args.backbone_aug_policy_setting) + "/" +  datasets[i] + "/" + str(args.probe_architecture)
        print(f'\nProbing dataset: {datasets[i]}')
        trainer = LinearProbeTrainer(
            dataset_base_pth=args.probe_datasets_base_pth,
            dataset_name=datasets[i],
            num_workers=(args.loader_workers+len(args.backbone_cuda_devices)-1)//len(args.backbone_cuda_devices),
            backbone_pth=args.backbone_pth,
            probe_pth=full_probe_pth,
            backbone_arch=args.backbone_architecture,
            probe_arch=args.probe_architecture,
            img_dims=args.img_dims,
            probe_layer=args.probe_layer,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            label_smoothing=args.probe_label_smoothing,
            epochs=args.probe_epochs,
            cuda_devices=args.probe_cuda_devices,
            wandb=wandb,
            use_wandb=args.use_wandb,
            device=device,
            seed=SEED)
        probe_res = trainer.train_probe(args.backbone_man_aug_setting)

        if args.use_wandb:
            accuracy_data = [
                [epoch + 1, value, series]
                for epoch, (train_acc, test_acc) in enumerate(
                    zip(probe_res['train_acc'], probe_res['test_acc'])
                )
                for value, series in zip([train_acc, test_acc], ["Train Accuracy", "Test Accuracy"])
            ]
            loss_data = [
                [epoch + 1, value, series]
                for epoch, (train_loss, test_loss) in enumerate(
                    zip(probe_res['train_loss'], probe_res['test_loss'])
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
                title=f"{datasets[i]} Probe Accuracy Over Epochs"
            )

            loss_chart = wandb.plot.line(
                table=loss_table,
                x="Epoch",
                y="Value",
                stroke="Series",  # Group lines by "Series" (Train Loss, Test Loss)
                title=f"{datasets[i]} Probe Loss Over Epochs"
            )
            wandb.log({
                f"{datasets[i]} Accuracy Chart": accuracy_chart,
                f"{datasets[i]} Loss Chart": loss_chart
            })
            results.append([
                i+1,
                args.backbone_architecture,
                str(args.backbone_man_aug_setting),
                str(args.backbone_aug_policy_setting),
                args.backbone_dataset_name,
                datasets[i],
                backbone_acc,
                probe_res['max_test_acc']
            ])

            df = pd.DataFrame(results, columns=[
                "Test_Num",
                "Backbone Architecture",
                "Manual Augmentation Setting",
                "Augmentation Policy Setting",
                "ID Dataset",
                "OOD Dataset",
                "Backbone ID max top-1 test acc",
                "Probe max top-1 test acc"
            ])  
            wandb.log({"Run Results": wandb.Table(dataframe=df)})
    print(f'\nProbed all datasets.')

    # Analysis

if __name__ == '__main__':
    if not torch.distributed.is_initialized():
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()