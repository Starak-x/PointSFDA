from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py
import random

class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, args):
        self.args = args
        self.dataset_path = '/workspace/dataset/PointCloudCompletion/CRN'
        self.class_choice = self.args.dataset.category
        self.split = self.args.dataset.split

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()] #(28974, 2048, 3)
        self.partial = data['incomplete_pcds'][()]#(28974, 2048, 3)
        self.labels = data['labels'][()]# (28974,)

        print("Load CRN "+self.split+" dataset")
        np.random.seed(0)
        self.cat_ordered_list = ['plane','cabinet','car','chair','lamp','sofa','table','watercraft']
        if self.class_choice =="all":
            self.index_list = np.array([])
            for cat in self.cat_ordered_list:
                cat_id = self.cat_ordered_list.index(cat.lower())
                self.index_list= np.append(self.index_list,np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ]))
                #self.class_number = np.append(self.class_number,len(self.index_list))
        else:
            cat_id = self.cat_ordered_list.index(self.class_choice.lower())
            self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id])

        #print(self.index_list.shape)
        #print(self.class_number)

    def __getitem__(self, index):

        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[int(full_idx)]) # fast alr
        label = self.labels[int(full_idx)]
        partial = torch.from_numpy(self.partial[int(full_idx)])
        #print(label)
        return gt, partial, label #full_idx

    def __len__(self):
        return len(self.index_list)

