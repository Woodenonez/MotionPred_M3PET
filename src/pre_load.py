import os, sys
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from net_module import loss_functions as loss_func
from network_manager import NetworkManager
from data_handle import data_handler as dh
from data_handle import dataset as ds

from util import utils_yaml

import pickle
from datetime import datetime

PRT_NAME = '[PRE]'

def check_device():
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count(),
              'Current 1:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
    else:
        print(f'CUDA not working! Pytorch: {torch.__version__}.')
        sys.exit(0)
    torch.cuda.empty_cache()

def load_config_fname(dataset_name, pred_range, mode):
    return f'{dataset_name.lower()}_{pred_range[0]}t{pred_range[1]}_{mode.lower()}.yml'

def load_param(root_dir, config_file, param_in_list=True, verbose=True):
    if param_in_list:
        param_list = utils_yaml.from_yaml_all(os.path.join(root_dir, 'Config/', config_file), vb=verbose)
        return {**param_list[0], **param_list[1], **param_list[2]}
    else:
        return utils_yaml.from_yaml(os.path.join(root_dir, 'Config/', config_file), vb=verbose)

def load_path(param, root_dir):
    save_path = None
    if param['model_path'] is not None:
        save_path = os.path.join(root_dir, param['model_path'])
    csv_path  = os.path.join(root_dir, param['label_path'])
    data_dir  = os.path.join(root_dir, param['data_path'])
    return save_path, csv_path, data_dir

def load_data(param, paths, transform, num_workers=0, T_range=None, ref_image_name=None, image_ext='png'):
    myDS = ds.ImageStackDataset(csv_path=paths[1], root_dir=paths[2], transform=transform,
                pred_offset_range=T_range, ref_image_name=ref_image_name, image_ext=image_ext)
    myDH = dh.DataHandler(myDS, batch_size=param['batch_size'], num_workers=num_workers)
    print(f'{PRT_NAME} Data prepared. #Samples(training, val):{myDH.get_num_data()}, #Batches:{myDH.get_num_batch()}')
    print(f'{PRT_NAME} Sample: \'image\':',myDS[0]['input'].shape,'\'label\':',myDS[0]['target'].shape)
    return myDS, myDH

def load_manager(param, Net, loss, verbose=True):
    net = Net(param['input_channel'], param['pred_len']) # in, out channels
    myNet = NetworkManager(net, loss, training_parameter=param, device=param['device'], verbose=verbose)
    myNet.build_Network()
    return myNet

def save_profile(manager, save_path='./'):
    dt = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    manager.plot_history_loss()
    plt.savefig(os.path.join(save_path, dt+'.png'), bbox_inches='tight')
    plt.close()
    loss_dict = {'loss':manager.Loss, 'val_loss':manager.Val_loss}
    with open(os.path.join(save_path, dt+'.pickle'), 'wb') as pf:
        pickle.dump(loss_dict, pf)

def main_train(root_dir, config_file, transform, Net, loss, num_workers:int, batch_size:int=None, 
               T_range:tuple=None, ref_image_name:str=None, image_ext='png'):
    ### Check and load
    check_device()
    param = load_param(root_dir, config_file)
    if batch_size is not None:
        param['batch_size'] = batch_size # replace the batch_size

    print(f'[Starting...] Model - {param["model_path"]}')
        
    paths = load_path(param, root_dir)
    _, myDH = load_data(param, paths, transform, num_workers, T_range, ref_image_name, image_ext)
    myNet = load_manager(param, Net, loss)

    ### Training
    start_time = time.time()
    myNet.train(myDH, myDH, param['batch_size'], param['epoch'])
    total_time = round((time.time()-start_time)/3600, 4)
    if (paths[0] is not None) & myNet.complete:
        torch.save(myNet.model.state_dict(), paths[0])
    nparams = sum(p.numel() for p in myNet.model.parameters() if p.requires_grad)
    print("\nTraining done: {} parameters. Cost time: {}h.".format(nparams, total_time))

    save_profile(myNet)

def main_test_pre(root_dir, config_file, transform, Net, ref_image_name:str=None, verbose=False):
    ### Check and load
    param = load_param(root_dir, config_file)
    paths = load_path(param, root_dir)
    myDS, myDH = load_data(param, paths, transform, ref_image_name=ref_image_name)
    if Net is not None:
        myNet = load_manager(param, Net, {})
        myNet.model.load_state_dict(torch.load(paths[0]))
        myNet.model.eval() # with BN layer, must run eval first
    else:
        myNet = None
    return myDS, myDH, myNet

def main_test(dataset, net, idx:int):
    try:
        ref = dataset[idx]['ref']
    except:
        ref = None
    img   = dataset[idx]['input']
    label = dataset[idx]['target']
    traj  = dataset[idx]['traj']
    index = dataset[idx]['index']
    pred = net.inference(img.unsqueeze(0))
    traj = torch.tensor(traj)
    return img, label, traj, index, pred.cpu(), ref