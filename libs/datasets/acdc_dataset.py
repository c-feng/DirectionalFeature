import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import json
import numpy as np
from PIL import Image
import h5py

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../'))
from utils.direct_field.df_cardia import direct_field
from libs.datasets import augment as standard_aug
from utils.direct_field.utils_df import class2dist


class AcdcDataset(Dataset):
    def __init__(self, data_list, df_used=False, joint_augment=None, augment=None, target_augment=None, df_norm=True, boundary=False):
        self.joint_augment = joint_augment
        self.augment = augment
        self.target_augment = target_augment
        self.data_list = data_list
        self.df_used = df_used
        self.df_norm = df_norm
        self.boundary = boundary
        
        with open(data_list, 'r') as f:
            self.data_infos = json.load(f)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self,index):
        img = h5py.File(self.data_infos[index],'r')['image']
        gt = h5py.File(self.data_infos[index],'r')['label']
        # print(np.unique(gt))
        img = np.array(img)[:,:,None].astype(np.float32)
        gt = np.array(gt)[:,:,None].astype(np.float32)
        # print(np.unique(gt))

        if self.joint_augment is not None:
            img, gt = self.joint_augment(img, gt)
        if self.augment is not None:
            img = self.augment(img)
        if self.target_augment is not None:
            gt = self.target_augment(gt)

        if self.df_used:
            gt_df = direct_field(gt.numpy()[0], norm=self.df_norm)
            gt_df = torch.from_numpy(gt_df)
        else:
            gt_df = None
        
        if self.boundary:
            dist_map = torch.from_numpy(class2dist(gt.numpy()[0], C=4))
        else:
            dist_map = None

        return img, gt, gt_df, dist_map


