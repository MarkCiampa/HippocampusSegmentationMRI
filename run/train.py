##########################
# Nicola Altini (2020)
# V-Net for Hippocampus Segmentation from MRI with PyTorch
##########################
# python run/train.py
# python run/train.py --epochs=NUM_EPOCHS --batch=BATCH_SIZE --workers=NUM_WORKERS --lr=LR
# python run/train.py --epochs=5 --batch=1 --net=unet

##########################
# Imports
##########################
import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold

##########################
# Local Imports
##########################
current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))

from run.utils import print_config, check_train_set, check_torch_loader, print_folder, train_val_split_config
from config.config import SemSegMRIConfig
from config.paths import logs_folder
from semseg.train import train_model, val_model
from semseg.data_loader import TorchIODataLoader3DTraining, TorchIODataLoader3DValidation
from semseg.weight import calc_weight
from models.vnet3d import VNet3D
from models.UNET3D import UNet
from models.FDUnet import FDUnet




def get_net(config):
    name = config.net
    assert name in ['unet', 'vnet', 'denseUnet'], "Network Name not valid or not supported! Use one of ['unet', 'vnet']"
    if name == 'vnet':
        return VNet3D(num_outs=config.num_outs, channels=config.num_channels)
    elif name == 'unet':
        return UNet(num_channels=1, num_outs=config.num_outs)
    elif name == 'denseUnet':
        return FDUnet(num_channels=1, num_outs=config.num_outs)


def run(config):

    ##########################
    # Check training set
    ##########################
    check_train_set(config)

    ##########################
    # Config
    ##########################
    print_config(config)
    net_folder = str(config.net)
    loss_folder = str(config.loss)
    
    
    ##########################
    # Create folders for training and test
    ##########################
    model_path = os.path.join(logs_folder,net_folder)
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, loss_folder)
    os.makedirs(model_path, exist_ok=True)
    model_path_gen = model_path
    

    ##########################
    # Check Torch DataLoader and Net
    ##########################
    check_torch_loader(config, check_net=False)

    ##########################
    # Training loop
    ##########################
    cuda_dev = torch.device('cuda')

    if config.do_crossval:
        ##########################
        # Training (cross-validation)
        ##########################
        multi_dices_crossval = list()
        mean_multi_dice_crossval = list()
        std_multi_dice_crossval = list()
        kf = KFold(n_splits=config.num_folders)

        if(config.loss == 'WCELoss'):
            print()
            weights = calc_weight(kf)
            print()
            print()
       

        
        for idx, (train_index, val_index) in enumerate(kf.split(config.train_images)):
            if(config.loss == 'WCELoss'):
                weight= weights[idx]

                weight=torch.tensor(weight)
                weight=weight.float()
                weight= weight.to(cuda_dev)
        

        

            print_folder(idx, train_index, val_index)
            config_crossval = train_val_split_config(config, train_index, val_index)

            ##########################
            # Training (cross-validation)
            ##########################
            net = get_net(config_crossval)
            config_crossval.lr = 0.01
            optimizer = optim.Adam(net.parameters(), lr=config_crossval.lr)
            train_data_loader_3D = TorchIODataLoader3DTraining(config_crossval)
            if(config.loss == 'WCELoss'):
                
                net = train_model(net, optimizer, train_data_loader_3D,
                              config_crossval,model_path, weights=weight, device=cuda_dev, logs_folder=logs_folder) 
            else:
                net = train_model(net, optimizer, train_data_loader_3D,
                                    config_crossval,model_path, device=cuda_dev, logs_folder=logs_folder)
            ##########################
            # Validation (cross-validation)
            ##########################
            val_data_loader_3D = TorchIODataLoader3DValidation(config_crossval)
            multi_dices, mean_multi_dice, std_multi_dice = val_model(net, val_data_loader_3D,
                                                                     config_crossval, device=cuda_dev)
            multi_dices_crossval.append(multi_dices)
            mean_multi_dice_crossval.append(mean_multi_dice)
            std_multi_dice_crossval.append(std_multi_dice)
            torch.save(net, os.path.join(model_path, "model_folder_{:d}.pt".format(idx)))

        ##########################
        # Saving Validation Results
        ##########################
        multi_dices_crossval_flatten = [item for sublist in multi_dices_crossval for item in sublist]
        mean_multi_dice_crossval_flatten = np.mean(multi_dices_crossval_flatten)
        std_multi_dice_crossval_flatten = np.std(multi_dices_crossval_flatten)
        print("Multi-Dice: {:.4f} +/- {:.4f}".format(mean_multi_dice_crossval_flatten, std_multi_dice_crossval_flatten))
        # Multi-Dice: 0.8728 +/- 0.0227

    ##########################
    # Training (full training set)
    ##########################
    net = get_net(config)
    if(config.loss == 'WCELoss'):
        weight= weights[len(weights)-1]
        weight=torch.tensor(weight)
        weight=weight.float()
        weight= weight.to(cuda_dev)
    config.lr = 0.01
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    train_data_loader_3D = TorchIODataLoader3DTraining(config)
    if(config.loss == 'WCELoss'):
                net = train_model(net, optimizer, train_data_loader_3D,
                              config,model_path,  weights=weight, device=cuda_dev, logs_folder=logs_folder) 
    else:
                net = train_model(net, optimizer, train_data_loader_3D,
                                    config, model_path, device=cuda_dev, logs_folder=logs_folder)

                                
    torch.save(net,os.path.join(model_path,"model.pt"))

    return model_path


############################
# MAIN
############################
if __name__ == "__main__":
    config = SemSegMRIConfig()

    parser = argparse.ArgumentParser(description="Run Training on Hippocampus Segmentation")
    parser.add_argument(
        "-e",
        "--epochs",
        default=config.epochs, type=int,
        help="Specify the number of epochs required for training"
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=config.batch_size, type=int,
        help="Specify the batch size"
    )
    parser.add_argument(
        "-v",
        "--val_epochs",
        default=config.val_epochs, type=int,
        help="Specify the number of validation epochs during training ** FOR FUTURE RELEASES **"
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=config.num_workers, type=int,
        help="Specify the number of workers"
    )
    parser.add_argument(
        "--net",
        default='vnet',
        help="Specify the network to use [unet | vnet] ** FOR FUTURE RELEASES **"
    )
    parser.add_argument(
        "--lr",
        default=config.lr, type=float,
        help="Learning Rate"
    )
    parser.add_argument(
        "--loss",
        default='MDLoss',
        help="Specify the loss function to user [MDLoss| CELoss | FLoss | MTLoss]"
    )
    parser.add_argument(
        "--folders",
        default=config.num_folders, type=int,
        help="Specify the number of folders for Cross Validation"
    )


    args = parser.parse_args()
    config.net = args.net
    config.epochs = args.epochs
    config.batch_size = args.batch
    config.val_epochs = args.val_epochs
    config.num_workers = args.workers
    config.lr = args.lr
    config.loss = args.loss
    config.num_folders = args.folders

    run(config)
