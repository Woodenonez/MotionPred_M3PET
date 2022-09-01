import os
import glob

import numpy as np
import pandas as pd
# from skimage import io
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset

from util import utils_np

'''
'''
class io(): # in case skimage not found
    @staticmethod
    def imread(path):
        tr = torchvision.transforms.PILToTensor()
        return tr(Image.open(path)).permute(1,2,0)

class ImageStackDataset(Dataset):
    def __init__(self, csv_path:str, root_dir:str, transform:torchvision.transforms=None, 
                 pred_offset_range:tuple=None, ref_image_name=None, image_ext='png'):
        '''
        Args:
            csv_path: Path to the CSV file with dataset info.
            root_dir: Directory with all image folders.
                      root_dir - video_folder - imgs
        '''
        super().__init__()
        self.info_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.tr = transform

        self.T_range = pred_offset_range
        self.ext = image_ext  # should not have '.'
        self.background = ref_image_name # if not None, use this as the background image

        csv_str = 'p' # in csv files, 'p' means position
        self.input_len = len([x for x in list(self.info_frame) if csv_str in x]) # length of input time step
        self.img_shape = self.check_img_shape(self.background)

    def __len__(self):
        return len(self.info_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_img = np.empty(shape=[self.img_shape[0],self.img_shape[1],0])
        info = self.info_frame.iloc[idx]
        traj = []
        
        if self.background is not None:
            img_name = self.background
        else:
            img_name = f'{info["index"]}.{self.ext}'
        img_path = os.path.join(self.root_dir, str(info['index']), img_name)
        image = self.togray(io.imread(img_path))
        for i in range(self.input_len):
            position = info['p{}'.format(i)]
            this_x = float(position.split('_')[0])
            this_y = float(position.split('_')[1])
            traj.append([this_x,this_y])

            # XXX
            # obj_map = utils_np.np_gaudist_map((this_x, this_y), np.zeros_like(image), sigmas=[10,10])
            obj_map = utils_np.np_dist_map((this_x, this_y), np.zeros_like(image))
            input_img = np.concatenate((input_img, obj_map[:,:,np.newaxis]), axis=2)
        input_img = np.concatenate((input_img, image[:,:,np.newaxis]), axis=2)
        
        if self.T_range is None:
            label_name_list = [x for x in list(self.info_frame) if 'T' in x]
        else:
            label_name_list = [f'T{x}' for x in range(self.T_range[0], self.T_range[1]+1)]
        label_list = list(info[label_name_list].values)
        label = [(float(x.split('_')[0]), float(x.split('_')[1])) for x in label_list]

        sample = {'input':input_img, 'target':np.array(label)} # label is TxDo
        if self.tr:
            sample = self.tr(sample)
        sample['index'] = info['index']
        sample['traj'] = traj
        sample['time'] = info['t']

        return sample

    def rescale_label(self, label, original_scale): # x,y & HxW
        current_scale = self.check_img_shape()
        rescale = (current_scale[0]/original_scale[0] , current_scale[1]/original_scale[1])
        return (label[0]*rescale[1], label[1]*rescale[0])

    def togray(self, image, normalize=True):
        if (len(image.shape)==2):
            return image
        elif (len(image.shape)==3) and (image.shape[2]==1):
            return image[:,:,0]
        else:
            image = image[:,:,:3] # ignore alpha
            img = image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
            if normalize:
                img /= 255
            return img

    def check_img_shape(self, img_name=None):
        info = self.info_frame.iloc[0]
        video_folder = str(info['index'])
        if img_name is None:
            img_name = video_folder + '.' + self.ext
        img_path = os.path.join(self.root_dir, video_folder, img_name)
        image = self.togray(io.imread(img_path))
        return image.shape

