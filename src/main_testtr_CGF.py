import os, sys
from pathlib import Path
from copy import deepcopy as copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from net_module.net import UNet, UNetLite, UNetLite_PELU
from data_handle import data_handler as dh
from data_handle import dataset as ds

import pre_load
from util import utils_test

print("Program: training\n")
if torch.cuda.is_available():
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
else:
    print(f'CUDA not working! Pytorch: {torch.__version__}.')
    sys.exit(0)
torch.cuda.empty_cache()

### XXX Back to energy map
def back2energy(x):
    x_original = copy(x)
    x[x_original>=1] *= -1
    x[x_original<1] = -torch.log(x[x_original<1])
    return x


### Config
root_dir = Path(__file__).parents[1]
config_file = 'sdd_1t12_train.yml'
ref_image_name = 'label.png'
# config_file = 'gcd_1t20_train.yml'
# ref_image_name = None #'label.png'
param = pre_load.load_param(root_dir, config_file, verbose=False)

composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = UNetLite

### Prepare
dataset, _, net = pre_load.main_test_pre(root_dir, config_file, composed, Net, ref_image_name=ref_image_name)

### Visualization option
idx_start = 10915#10650 # 10915 # 17850 # 250
idx_end = len(dataset)
pause_time = 0

### Visualize
fig, axes = plt.subplots(2,2)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    print(f'\r{idx}', end='')

    [ax.cla() for ax in axes.ravel()]
    
    img, label, traj, index, e_grid, ref = pre_load.main_test(dataset, net, idx=idx)
    try:
        prob_map = net.convert_grid2prob(e_grid.clone(), threshold=0.1, temperature=1)
    except:
        prob_map = e_grid.clone()
        prob_map[prob_map<0.011] = 0
        e_grid = back2energy(e_grid)
    pred_traj = utils_test.get_traj_from_pmap(prob_map)

    if ref is None:
        ref = img[-1,:,:]

    mu_list_list = []
    std_list_list = []
    conf_list_list = []
    traj_samples = net.gen_samples(prob_map, num_samples=50, replacement=True)
    for i in range(prob_map.shape[1]):
        # goal_samples = utils_func.ynet_TTST(prob_map[:,i,:,:].unsqueeze(1), 20)
        # clusters = utils_test.fit_DBSCAN(goal_samples[:,0,0,:].numpy(), eps=5, min_sample=3)

        # prob_x, prob_y = traj_samples[0,i,:,0], traj_samples[0,i,:,1]
        # confidence = prob_map[0, i, prob_y.long(), prob_x.long()]

        clusters = net.fit_DBSCAN(traj_samples[0,i,:].numpy(), eps=10, min_sample=5)
        mu_list, std_list = net.fit_cluster2gaussian(clusters)

        conf_list = []
        for mu in mu_list:
            conf_list.append(prob_map[0, i, int(mu[1]), int(mu[0])].item()+1e-9)
        conf_list = [round(x/sum(conf_list),2) for x in conf_list]

        mu_list_list.append(mu_list)
        std_list_list.append(std_list)
        conf_list_list.append(conf_list)

    alpha_list = np.linspace(start=1, stop=0.5, num=len(mu_list_list)).tolist()

    utils_test.plot_on_ref([axes[0,0], axes[1,0], axes[1,1]], ref, traj, label,
                            e_grid[0,-1], prob_map[0,-1], traj_samples)

    axes[0,1].imshow(ref, cmap='gray')
    axes[0,1].plot(traj[-1,0], traj[-1,1], 'ro', label='current')
    axes[0,1].plot(traj[:-1,0], traj[:-1,1], 'r.') # past
    axes[0,1].plot(label[:,0], label[:,1], 'r-', label="ground truth")
    for mu_list, std_list, conf_list, alpha in zip(mu_list_list, std_list_list, conf_list_list, alpha_list):
        utils_test.plot_Gaussian_ellipses(axes[0,1], mu_list, std_list, conf_list, factor=3, expand=5, alpha=alpha)

    ### Zoom in
    # for ax in [axes[0,1]]:
    #     ax.set_xlim(min(pred_traj[:,0].min(), label[:,0].min().item())-100, max(pred_traj[:,0].max(), label[:,0].max().item())+100)
    #     ax.set_ylim(max(pred_traj[:,1].max(), label[:,1].max().item())+100, min(pred_traj[:,1].min(), label[:,1].min().item())-100)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    
    plt.draw()
    if pause_time==0:
        plt.pause(0.01)
        while not plt.waitforbuttonpress():  # XXX press a button to continue
            pass
    else:
        plt.pause(pause_time)

plt.show()
