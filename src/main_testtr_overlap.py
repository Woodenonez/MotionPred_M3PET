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

# import warnings
# warnings.filterwarnings("error")

print("Program: training\n")
if torch.cuda.is_available():
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
else:
    print(f'CUDA not working! Pytorch: {torch.__version__}.')
    sys.exit(0)
torch.cuda.empty_cache()

### Config
root_dir = Path(__file__).parents[1]
config_file = 'msmd_1t10_test.yml'
param = pre_load.load_param(root_dir, config_file, verbose=False)

data_from_zip = False
composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = UNet

### Prepare
dataset, _, net = pre_load.main_test_pre(root_dir, config_file, Dataset, data_from_zip, composed, Net)

### Visualization option
idx_start = 160 # 250
idx_end = len(dataset)
pause_time = 0.1

### Visualize
fig, axes = plt.subplots(1,3)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    print(f'\r{idx}   ', end='')

    [ax.cla() for ax in axes.ravel()]
    
    img, label, traj, index, e_grid, ref = pre_load.main_test(dataset, net, idx=idx)
    prob_map = loss_func.convert_grid2prob(e_grid.clone(), threshold=0.1, temperature=0.5)
    if ref is None:
        ref = img[-1,:,:]

    ### Normalize
    # prob_map = prob_map/torch.amax(prob_map, dim=(2,3))[:,:,None,None]

    ax1, ax2, ax3 = axes

    if param['cell_width'] > 2:
        x_grid = np.arange(0, param['x_max_px']+1, param['cell_width'])
        y_grid = np.arange(0, param['y_max_px']+1, param['cell_width'])
        ax1.set_xticks(x_grid)
        ax1.set_yticks(y_grid)
        ax1.grid(linestyle='-')

    ax1.imshow(ref, cmap='gray')
    ax1.plot(traj[-1,0], traj[-1,1], 'ko', label='current')
    ax1.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    ax1.plot(label[:,0], label[:,1], 'r-', linewidth=4, label="ground truth")
    # ax1.plot(pred_traj[:,0], pred_traj[:,1], 'gx', label="pred")
    ax1.legend(prop={'size': 14}, loc='best')
    ax1.set_aspect('equal', 'box')

    ax2.imshow(torch.sum(e_grid[0,:], dim=0), cmap='gray')
    ax2.plot(label[:,0], label[:,1], 'r.')

    ax3.imshow(torch.clamp(torch.sum(prob_map[0,:], dim=0), max=1), cmap='gray')
    ax3.plot(label[:,0], label[:,1], 'r-')

    ax1.set_title('Real world')
    ax2.set_title('Energy grid')
    ax3.set_title('Probability map')
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
