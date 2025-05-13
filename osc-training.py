import numpy as np # Successfully installed numpy-1.23.5
import pandas as pd # Successfully installed pandas-1.3.5
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import torch
import sys
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import json
from models.models import GATWithJK
from preprocessing import graph_creation
from training_utils import PyTorchTrainer, PyTorchDistillationTrainer, DistillationTrainer
import torch.profiler

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):

    # add hydra instantiation to the script
    config_dict = OmegaConf.to_container(config, resolve=True)
    print(config_dict)

    # make device the first line
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Model is using device: {device}')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    root_folders = {'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
                    'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
                    'set_01' : r"datasets/can-train-and-test-v1.5/set_01",
                    'set_02' : r"datasets/can-train-and-test-v1.5/set_02",
                    'set_03' : r"datasets/can-train-and-test-v1.5/set_03",
                    'set_04' : r"datasets/can-train-and-test-v1.5/set_04",
    }

    KEY = config_dict['root_folder']
    root_folder = root_folders[KEY]
    print(f"Root folder: {root_folder}")

    dataset = graph_creation(root_folder)
    print(f"Number of graphs: {len(dataset)}")

    for data in dataset:
        assert not torch.isnan(data.x).any(), "Dataset contains NaN values!"
        assert not torch.isinf(data.x).any(), "Dataset contains Inf values!"
    
    # hyperparameters from yaml file
    DATASIZE = config_dict['datasize']
    EPOCHS = config_dict['epochs']
    LR = config_dict['lr']
    BATCH_SIZE = config_dict['batch_size']
    TRAIN_RATIO = config_dict['train_ratio']
    USE_FOCAL_LOSS = config_dict['use_focal_loss']  # Read from the YAML config

    print("Size of the total dataset: ", len(dataset))
   

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator1)
    print('Size of DATASIZE: ', DATASIZE)
    print('Size of Training dataset: ', len(train_dataset))
    print('Size of Testing dataset: ', len(test_dataset))
    

    if DATASIZE < 1.0:
        subset_size = int(len(train_dataset) * DATASIZE)  # Fraction of the total dataset
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print('Size of Training dataloader: ', len(train_loader))
    print('Size of Testing dataloader: ', len(test_loader))
    print('Size of Training dataloader (samples): ', len(train_loader.dataset))
    print('Size of Testing dataloader (samples): ', len(test_loader.dataset))

    # Knowledge Distillation Scenario
    teacher_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=5, heads=8).to(device)
    student_model = GATWithJK(in_channels=10, hidden_channels=32, out_channels=1, num_layers=2, heads=4).to(device)

    # Initialize the DistillationTrainer
    trainer = DistillationTrainer(
        teacher=teacher_model,
        student=student_model,
        device=device,
        teacher_epochs=EPOCHS,  # Train teacher for 50 epochs
        student_epochs=EPOCHS,  # Train student for 50 epochs
        distill_alpha=0.5,      # Weight for distillation loss
        warmup_epochs=5,        # Warmup epochs for student training
        lr=LR ,
        use_focal_loss=USE_FOCAL_LOSS,  # Use Focal Loss for student training
    )

    
    # Train teacher first, then student
    print("Starting sequential training...")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            trainer.train_sequential(train_loader, test_loader)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # trainer.train_sequential(train_loader, test_loader)

    # Define the folder to save the models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

    teacher_model_filename = os.path.join(save_folder, f'best_teacher_model_{KEY}.pth')
    torch.save(trainer.best_teacher_model, teacher_model_filename)
    print(f"Best teacher model saved as '{teacher_model_filename}'.")

    student_model_filename = os.path.join(save_folder, f'final_student_model_{KEY}.pth')
    torch.save(student_model.state_dict(), student_model_filename)
    print(f"Final student model saved as '{student_model_filename}'.")
    
    # Save performance metrics
    metrics = {
        "teacher_metrics": trainer.teacher_metrics,  # Assuming trainer tracks teacher metrics
        "student_metrics": trainer.student_metrics   # Assuming trainer tracks student metrics
    }
    print(metrics)
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Training metrics saved as 'training_metrics.json'.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")