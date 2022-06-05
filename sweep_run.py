import wandb
from NN import NeuralNet
import torch
from dataset import ContourDataset
from trainer import train
from torch.utils.data import DataLoader
from torch import autograd
import numpy as np
import random 
import wandb

# Seed
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


id = 'o0jgljrv'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run():
    with wandb.init(project="2d_unet", entity="master-thesis-ntnu", config=dict) as run:
        config = wandb.config

        df_loss_value = config.df_loss_value

        # Hyper-parameters
        num_epochs = 5
        batch_size = 8  
        learning_rate = 0.001 
       
        use_existing_model = True
        # Dataset
        train_dataset = ContourDataset()
        test_dataset = ContourDataset(split="test")
        validation_dataset = ContourDataset(split="validation")
        # Data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=3, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)
        
        model = NeuralNet().to(device)
        if use_existing_model:
            model.load_state_dict(torch.load("model_check_point_best_l1.pth")["state_dict"])
        #with autograd.detect_anomaly():
        wandb.init(project="2d_unet", entity="master-thesis-ntnu",config=dict)
        wandb.watch(model, log_freq=5)
        model = train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, df_loss_value, learning_rate=learning_rate, device=device)
        
        # Save model
        config = {
            "state_dict": model.state_dict()
        }
        torch.save(config, "model_end.pth")

if __name__ == "__main__":
    wandb.agent(id, project="2d_unet", entity="master-thesis-ntnu", function=run, count=50)