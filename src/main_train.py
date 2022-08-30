import os, sys
from pathlib import Path

import torch
import torchvision

from net_module import loss_functions as loss_func
from net_module.net import UNet, UNetLite
from data_handle import data_handler as dh

import pre_load

print("Program: training\n")

DATASET = 'GCD'
MODE = 'TRAIN'
PRED_RANGE = (1,20) # E.g. (1,10) means 1 to 10

BATCH_SIZE = 2

### Config
root_dir = Path(__file__).parents[1]

# loss = {'loss': torch.nn.BCEWithLogitsLoss(), 'metric': loss_func.loss_mae}
loss = {'loss': loss_func.loss_enll, 'metric': loss_func.loss_mae}
config_file = pre_load.load_config_fname(DATASET, PRED_RANGE, MODE)
composed = torchvision.transforms.Compose([dh.ToTensor()])
Net = UNetLite

### Training
pre_load.main_train(root_dir, config_file, Net=Net, transform=composed, loss=loss, num_workers=0, 
                    batch_size=BATCH_SIZE, T_range=PRED_RANGE, ref_image_name=None)

