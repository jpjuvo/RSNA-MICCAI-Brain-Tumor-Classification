import glob
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2

class BraTS2021(Dataset):
    """
    Dataset for BraTS2021 challenge - Includes tasks 1 and 2
    """

    def __init__(self, mode, 
                 npy_fns_list,
                 label_list=[],
                 augmentations=None,
                 volume_normalize=True):
        """
        :param mode: 'train','val','test'
        :param npy_fns_list: list of numpy array paths
        :param label_list: list of binary label integers
        :param augmentations: 3D augmentations
        :param volume_normalize: z-score normalize each channel in sample
        """
        self.mode = mode
        self.augmentations = augmentations
        self.volume_normalize = volume_normalize
        
        self.seg_fn_list = []
        self.fn_list = npy_fns_list
        self.label_list = label_list
        
        if self.mode != "test":
            fn_list_with_seg = []
            label_list_with_seg = []
            for fn, lbl in zip(npy_fns_list, label_list):
                fn_seg = str(fn).replace('.npy', '_seg.npy')
                if os.path.exists(fn_seg) and os.path.exists(fn):
                    fn_list_with_seg.append(fn)
                    self.seg_fn_list.append(fn_seg)
                    label_list_with_seg.append(int(lbl))
            
            # update fn list and exclude ones that don't have seg map
            self.fn_list = fn_list_with_seg
            self.label_list = label_list_with_seg
            assert len(self.fn_list) == len(self.seg_fn_list) == len(self.label_list)
        
        ########################
        #### Sanity checks #####
        
        assert len(self.fn_list) > 0
        
        # Check one sample
        one_sample = np.load(self.fn_list[0])
        assert len(one_sample.shape) == 4
        d,x,y,c = one_sample.shape
        self.full_vol_dim = (x,y,d)
        self.channels = c
        
        # Check one seg map
        if len(self.seg_fn_list) > 0:
            one_seg_sample = np.load(self.seg_fn_list[0])
            assert len(one_seg_sample.shape) == 3
            seg_d,seg_x,seg_y = one_seg_sample.shape
            assert seg_d == d
            assert seg_x == x
            assert seg_y == y
            
        #### Sanity checks #####
        ########################
        
        # shuffle samples
        if self.mode != "test":
            all_lists = list(zip(self.fn_list, self.seg_fn_list, self.label_list))
            random.shuffle(all_lists)
            self.fn_list, self.seg_fn_list, self.label_list = zip(*all_lists)
        
    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, index):
        sample = np.load(self.fn_list[index]).astype(np.float32)
        out_dict = {
            'BraTSID' : os.path.basename(self.fn_list[index]).split('.')[0],
            'image' : sample
        }
        
        if self.mode != "test":
            seg = np.load(self.seg_fn_list[index])
            # set seg to binary values
            seg = (seg > 0).astype(np.float32)
            lbl = self.label_list[index]
            out_dict['segmentation'] = seg 
            out_dict['label'] = lbl
            
            if self.augmentations:
                out_dict = self.augmentations(out_dict)
        
        # z-score norm each channel - done after augmentations
        if self.volume_normalize:
            sample = out_dict['image'].copy()
            sample_mean = np.mean(sample, axis=tuple([0,1,2]))
            sample_std = np.std(sample, axis=tuple([0,1,2])) + 1e-6
            sample = (sample - sample_mean) / sample_std
            out_dict['image'] = sample
            out_dict['mean'] = sample_mean
            out_dict['std'] = sample_std
        else:
            out_dict['mean'] = np.array([0 for _ in range(sample.shape[3])])
            out_dict['std'] = np.array([1. for _ in range(sample.shape[3])])
            
        return out_dict