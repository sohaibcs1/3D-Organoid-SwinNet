import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from train.utils import train
from model.3D_Organoid_SwinNet import 3D_Organoid_SwinNet
from loader.loader import data_loaders

def main():
    # Load hyperparameters from JSON file
    with open('configs.json', 'r') as config_file:
        config = json.load(config_file)

    # dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, train_loader = data_loaders(config["data_dir"], num_samples=config["num_samples"], device=device)

    # define the model
    model = 3D_Organoid_SwinNet(
        img_size=(config["input_size"], config["input_size"], config["input_size"]),
        in_channels=config["input_channels"],
        out_channels=config["num_classes"],
        feature_size=config["feature_size"],
        use_checkpoint=config["use_checkpoint"],
    ).to(device)

    ## train loop 
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    while global_step < config["max_iterations"]:
        global_step, dice_val_best, global_step_best = train(model, global_step, train_loader, val_loader,config, dice_val_best, global_step_best)

if __name__ == "__main__":
    main()
