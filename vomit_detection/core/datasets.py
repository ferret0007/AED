import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import pickle

import csv
import argparse

from PIL import Image

class MouseDataset(data.Dataset):
    def __init__(self, is_test=False, params={}, data_path=None):


        self.init_seed = False
        self.is_test = is_test
        self.data_list = [] # save all data for fast fetching
        self.params = params
        self.cache_data = {}
        self.data_path = data_path

        self.load_data(data_path)
        print("[Frame num is {}. In total {} batch.]".format(self.params["frame_num"], len(self.data_list)))
        
        # self.data_list[0] = self.data_list[0][:100]
        # self.data_list[1] = self.data_list[1][:100]

        if self.params["cache_data"]:
            print("[Start caching data]")
            num = 0
            for dir in self.data_list[0]+self.data_list[1]:
                for file in dir:
                    self.cache_data[file] = np.array(Image.open(file))
                    num += 1
                    if num % 10000 == 0:
                        print(num)

    def load_data(self, data_root):

        img_files = sorted(os.listdir(data_root))
        batch_num = len(img_files) // self.params["frame_num"]

        for batch_id in range(batch_num):
            this_list = []
            for idx in range(30):
                this_list.append(os.path.join(data_root, "{:05d}.jpg".format(batch_id*self.params["frame_num"]+idx)))
            self.data_list.append(this_list)

    def __getitem__(self, index):
        if not self.init_seed:
            # workers get different random seed
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        files = self.data_list[index]
        num_files = len(files)
        start_idx = 0
        files = files[start_idx:start_idx+self.params["frame_num"]]
        name = files[0]
        if self.params["cache_data"]:
            files = [torch.from_numpy(self.cache_data[_file]).float().permute(2,0,1) for _file in files]
        else:
            files = [torch.from_numpy(np.array(Image.open(_file))).float().permute(2,0,1) for _file in files]
        files = [_file/127.5-1.0 for _file in files]
        files = torch.stack(files, dim=1)
        
        return files, index

    def __len__(self):
        return len(self.data_list)

def fetch_dataloader(args, is_test=False, drop_last=True, data_path=None):
    """ Create the data loader for the corresponding trainign set """

    params = {"frame_num": args.frame_num, "data_balance": args.data_balance, "cache_data": args.cache_data}
    
    train_dataset = MouseDataset(is_test=is_test, params=params, data_path=data_path)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=not is_test, num_workers=8, drop_last=drop_last)

    print('Testing with %d data' % len(train_dataset))
    return train_loader
        
