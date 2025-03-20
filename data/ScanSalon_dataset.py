from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import random
import glob
from plyfile import PlyData
from utils.pc_transform import swap_axis

class RandomSamplePoints(object):
    def __init__(self, npoints):
        self.n_points = npoints

    def __call__(self, pcd):
        choice = np.random.permutation(pcd.shape[0])
        pcd = pcd[choice[:self.n_points]]

        if pcd.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - pcd.shape[0], 3))
            pcd = np.concatenate([pcd, zeros])

        return pcd

# def normalize(pcd, get_arg=False, center=None, max_scale=None):
#     if center is None or max_scale is None:
#         maxs = np.max(pcd, 0, keepdims=True)
#         mins = np.min(pcd, 0, keepdims=True)
#         center = (maxs+mins)/2
#         scale = (maxs-mins)/2
#         max_scale = np.max(scale)
#     pcd = pcd - center
#     pcd = pcd / max_scale *0.5
#     if get_arg:
#         return pcd, center, max_scale
#     else:
#         return pcd

def pc_norm(pcd, get_arg=False, center=None, max_scale=None):
    """ pc: NxC, return NxC """
    if center is None or max_scale is None:
        center = np.mean(pcd, axis=0)
        max_scale = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
    pcd = pcd - center

    pcd = pcd / max_scale * 0.5
    if get_arg:
        return pcd, center, max_scale
    else:
        return pcd

class ScanSalonDataset(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, args):
        self.args = args
        self.dataset_path = '/workspace/dataset/PointCloudCompletion/ScanSalon'
        self.class_choice = self.args.dataset.category
        #self.dataset_path = self.args.DATASETS.SCANSALON.DATA_PATH
        #self.class_choice = self.args.DATASET.CLASS
        self.partials = []
        self.gts = []
        self.labels = []
        self.n_points=2048
        #self.split = self.args.DATASET.SPLIT


        #pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
       # data = h5py.File(pathname, 'r')
       #  self.gt = data['complete_pcds'][()]
       #  self.partial = data['incomplete_pcds'][()]
       #  self.labels = data['labels'][()]
        self.class_number = []
        np.random.seed(0)
        self.cat_ordered_list = ['car','desk','sofa','chair','lamp']
        if self.class_choice == "all":
            #self.index_list = np.array([])
            for cat in self.cat_ordered_list:
                cat_id = self.cat_ordered_list.index(cat.lower())
                partial_path = self.dataset_path+'/partials/'+ cat
                partial_list = sorted(glob.glob(partial_path + '/*'))

                gt_path = self.dataset_path + '/gts/' + cat
                gt_list = sorted(glob.glob(gt_path + '/*'))
                total_num = len(partial_list)
                for num in range(total_num):

                    partial_ply = PlyData.read(partial_list[num])
                    partial = np.array([partial_ply['vertex']['x'], partial_ply['vertex']['y'], partial_ply['vertex']['z']])
                    partial = np.transpose(partial,(1,0))
                    gt_ply = PlyData.read(gt_list[num])
                    gt = np.array([gt_ply['vertex']['x'], gt_ply['vertex']['y'], gt_ply['vertex']['z']])
                    gt = np.transpose(gt, (1, 0))#(2048,3)

                    self.partials.append(partial)
                    self.gts.append(gt)
                    self.labels.append(cat_id)

                # self.index_list= np.append(self.index_list,np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ]))
                # self.class_number = np.append(self.class_number,len(self.index_list))
        else:
            cat_id = self.cat_ordered_list.index(self.class_choice.lower())
            partial_path = self.dataset_path + '/partials/' + self.class_choice
            partial_list = sorted(glob.glob(partial_path + '/*'))
            gt_path = self.dataset_path + '/gts/' + self.class_choice
            gt_list = sorted(glob.glob(gt_path + '/*'))
            total_num = len(partial_list)
            for num in range(total_num):
                partial_ply = PlyData.read(partial_list[num])
                partial = np.array([partial_ply['vertex']['x'], partial_ply['vertex']['y'], partial_ply['vertex']['z']])
                partial = np.transpose(partial, (1, 0))
                self.partials.append(partial)
                gt_ply = PlyData.read(gt_list[num])
                gt = np.array([gt_ply['vertex']['x'], gt_ply['vertex']['y'], gt_ply['vertex']['z']])
                gt = np.transpose(gt, (1, 0))
                self.gts.append(gt)
                self.labels.append(cat_id)
        # self.partials = [swap_axis(itm, swap_mode='n210') for itm in self.partials]
        # self.gt = [swap_axis(itm, swap_mode='n210') for itm in self.gts]
        #print(len(self.partials))



    def RandomSamplePoints(self,pcd):
        choice = np.random.permutation(pcd.shape[0])
        pcd = pcd[choice[:self.n_points]]

        if pcd.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - pcd.shape[0], 3))
            pcd = np.concatenate([pcd, zeros])

        return pcd

    def __getitem__(self, index):
        partial = self.RandomSamplePoints(self.partials[index])
        gt = self.gts[index]
        label = self.labels[index]
        #print(gt.shape,partial.shape,label)
        # gt, center, max_scale = pc_norm(gt, get_arg=True)
        # partial = pc_norm(partial, center=center, max_scale=max_scale)

        return (gt, partial, int(label))

    def __len__(self):
        return len(self.gts)


