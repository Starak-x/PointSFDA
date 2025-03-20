# -*- coding: utf-8 -*-
# @Author: XP

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#
__C.DATASETS = edict()
__C.DATASETS.CRN = edict()
__C.DATASETS.CRN.DATA_PATH = '/mnt/star/datasets/CRN'
__C.DATASETS.USSPA = edict()
__C.DATASETS.USSPA.SHAPANET = '/home/starak/PointCompletion/dataset/USSPA/RealComShapeNetData/shapenet_data.lmdb'
__C.DATASETS.USSPA.TRAIN = '/home/starak/PointCompletion/dataset/USSPA/RealComData/realcom_data_train.lmdb'
__C.DATASETS.USSPA.VALID = '/home/starak/PointCompletion/dataset/USSPA/RealComData/realcom_data_test.lmdb'
__C.DATASETS.SCANSALON = edict()
__C.DATASETS.SCANSALON.DATA_PATH = '/mnt/star/datasets/ScanSalon'



#
# Dataset
__C.DATASET = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars
__C.DATASET.VIRTUAL_DATASET = 'CRN'
__C.DATASET.REAL_DATASET = '3D_FUTURE'  # 3D_FUTURE  ModelNet
__C.DATASET.SPLIT = 'train'
__C.DATASET.CLASS = 'all'

#
# Constants
#
__C.CONST = edict()

__C.CONST.NUM_WORKERS = 8
__C.CONST.N_INPUT_POINTS = 2048

#
# Directories
#

__C.DIR = edict()
__C.DIR.OUT_PATH = './outpath_snow'
__C.CONST.DEVICE = '0'
# __C.CONST.WEIGHTS                                = './outpath_p2c/2023-10-10T19:16:08.674376/checkpoints/ckpt-best.pth'
#__C.CONST.WEIGHTS                                   = './outpath_distill/2023-12-06T14/checkpoints/ckpt-epoch-250.pth'
# __C.CONST.SOURCE_WEIGHTS                                = './outpath_distill/10-23T19-3090/ckpt-best.pth' # 'ckpt-best.pth'  # specify a path to run test and inference
#__C.CONST.WEIGHTS                                = './outpath_distill/2023-12-21partial-chair/checkpoints/ckpt-best.pth'
# __C.CONST.ADAPTER_WEIGHTS                        = './outpath_distill/11-7-3090without_tanh/adapter-best.pth'
#__C.CONST.SOURCE_WEIGHTS                        = './outpath_snowflakenet/checkpoints/2023-12-11T17:31:33.157755/ckpt-best-source.pth'
#__C.CONST.SOURCE_WEIGHTS                          = './outpath_snowflakenet/12-21-4090/checkpoints/ckpt-best.pth'
#__C.CONST.SOURCE_WEIGHTS                                = './outpath_distill/2023-12-21partial-chair/checkpoints/ckpt-best-source.pth'
#
# Memcached
#
__C.MEMCACHED = edict()
__C.MEMCACHED.ENABLED = False
__C.MEMCACHED.LIBRARY_PATH = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.N_POINTS = 2048

#
# Train
#
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 8  # 16
__C.TRAIN.N_EPOCHS = 500
__C.TRAIN.SAVE_FREQ = 50
__C.TRAIN.LEARNING_RATE = 0.00004#0.00004  # 0.0001 #0.001 0.0005 #0.0001 for PCN
__C.TRAIN.LR_MILESTONES = [50, 100, 250, 300, 350]  # [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP = 40
__C.TRAIN.WARMUP_STEPS = 200
__C.TRAIN.GAMMA = .5  # .5
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.WEIGHT_DECAY = 0
__C.TRAIN.ADAPTER_LEARNING_RATE = 0.0001



__C.TEST = edict()
__C.TEST.METRIC_NAME = 'ChamferDistance'
