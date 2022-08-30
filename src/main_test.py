import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from net_module.net import UNet
from data_handle import data_handler as dh
from data_handle import dataset as ds
from net_module import loss_functions as loss_func

import pre_load
from util import utils_test

print("Program: training\n")
if torch.cuda.is_available():
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
else:
    print(f'CUDA not working! Pytorch: {torch.__version__}.')
    sys.exit(0)
torch.cuda.empty_cache()

### Config
root_dir = Path(__file__).parents[1]
config_file = 'sid_20_test.yml'
param = pre_load.load_param(root_dir, config_file, verbose=False)

data_from_zip = False
composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = UNet

### Prepare
dataset, _, net = pre_load.main_test_pre(root_dir, config_file, Dataset, data_from_zip, composed, Net)

### Visualization option
idx_start = 0 # 250
idx_end = len(dataset)
pause_time = 0.1

### Visualize
fig, axes = plt.subplots(1,3)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    print(idx)

    [ax.cla() for ax in axes]
    
    img, label, traj, index, e_grid, ref = pre_load.main_test(dataset, net, idx=idx)
    # e_min  = np.min(e_grid)
    # e_max  = np.max(e_grid)
    # e_grid = (e_grid-e_min)/(e_max-e_min)
    prob_map = loss_func.convert_grid2prob(e_grid.clone(), threshold=0.1, temperature=0.5)

    # utils_test.plot_on_sdd(axes, img[-2,:,:], px_idx, cell_idx, traj, e_grid[0,:], prob_map[0,:])

    for ax in axes.ravel():
        ax.plot(traj[-1,0], traj[-1,1], 'ko', label='current')

    axes[0].imshow(img[-1], cmap='gray')
    axes[0].plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    axes[0].plot(label[:,0], label[:,1], 'bo', label="ground truth")
    axes[0].legend(prop={'size': 14}, loc='best')
    axes[0].set_aspect('equal', 'box')

    axes[1].imshow(e_grid[0,0,:], cmap='gray')
    axes[1].plot(label[0,0], label[0,1], 'rx')

    axes[2].imshow(prob_map[0,0,:], cmap='gray')
    axes[2].plot(label[0,0], label[0,1], 'rx')

    if param['cell_width']>2:
        x_grid = np.arange(0, param['x_max_px']+1, param['cell_width'])
        y_grid = np.arange(0, param['y_max_px']+1, param['cell_width'])
        axes[0].set_xticks(x_grid)
        axes[0].set_yticks(y_grid)
        axes[0].grid(linestyle='-')
        
    # boundary_coords, obstacle_list, _ = return_Map(index)
    # graph = Graph(boundary_coords, obstacle_list)
    # graph.plot_map(ax1, clean=1)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    
    if pause_time==0:
        plt.pause(0.1)
        input()
    else:
        plt.pause(pause_time)

plt.show()
