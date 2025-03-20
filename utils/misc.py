import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
from pointnet2_ops.pointnet2_utils import gather_operation,furthest_point_sample

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]
    
def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

class EMA(object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """

    def __init__(self, teacher_net, student_net, alpha=0.999):
        self.teacher_params = list(teacher_net.parameters())
        self.student_params = list(student_net.parameters())
        self.alpha = alpha
        # for p, src_p in zip(self.target_params, self.source_params):
        #     p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, stu_p in zip(self.teacher_params, self.student_params):
            p.data.mul_(self.alpha)
            p.data.add_(stu_p.data * one_minus_alpha)


def split(pcd, view, x_center, y_center, z_center):
    if view == 0:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] < x_center,
                                                       pcd[:, 1] < y_center),
                                     pcd[:, 2] < z_center)
        pcd = pcd[~data_pcd, :]
    elif view == 1:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] < x_center,
                                                       pcd[:, 1] < y_center),
                                     pcd[:, 2] > z_center)
        pcd = pcd[~data_pcd, :]
    elif view == 2:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] < x_center,
                                                       pcd[:, 1] > y_center),
                                     pcd[:, 2] < z_center)
        pcd = pcd[~data_pcd, :]
    elif view == 3:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] < x_center,
                                                       pcd[:, 1] > y_center),
                                     pcd[:, 2] > z_center)
        pcd = pcd[~data_pcd, :]
    elif view == 4:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] > x_center,
                                                       pcd[:, 1] < y_center),
                                     pcd[:, 2] < z_center)
        pcd = pcd[~data_pcd, :]
    elif view == 5:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] > x_center,
                                                       pcd[:, 1] < y_center),
                                     pcd[:, 2] > z_center)
        pcd = pcd[~data_pcd, :]
    elif view == 6:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] > x_center,
                                                       pcd[:, 1] > y_center),
                                     pcd[:, 2] < z_center)
        pcd = pcd[~data_pcd, :]
    elif view == 7:
        data_pcd = torch.logical_and(torch.logical_and(pcd[:, 0] > x_center,
                                                       pcd[:, 1] > y_center),
                                     pcd[:, 2] > z_center)
        pcd = pcd[~data_pcd, :]

    return pcd
# def mask_aug(pcds):
#     bs, num, _ = pcds.shape
#     device = pcds.device
#     pcd1s = []
#     pcd2s = []
#     for i in range(bs):

#         pcd1 = pcds[i]
#         pcd2 = pcds[i]
#         [x_min, y_min, z_min], idx = torch.min(pcd1[:, :3], axis=0)
#         [x_max, y_max, z_max], idx = torch.max(pcd1[:, :3], axis=0)

#         x_center = (x_max + x_min) / 2.
#         y_center = (y_max + y_min) / 2.
#         z_center = (z_max + z_min) / 2.
#         # x_center, y_center, z_center = torch.mean(corners, 0)

#         view = torch.randint(0, 8, (2,))
#         while view[0] == view[1]:
#             view = torch.randint(0, 8, (2,))
      

#         pcd1 = split(pcd1, view[0], x_center, y_center, z_center)
#         # pcd2 = split(pcd2,view[0],x_center,y_center,z_center)
#         pcd2 = split(pcd2,view[1],x_center,y_center,z_center)
#         pcd1 = resample_pcd(pcd1, num)
#         pcd2 = resample_pcd(pcd2, num)
#         pcd1s.append(pcd1.unsqueeze(0))
#         pcd2s.append(pcd2.unsqueeze(0))

#     pcd1s = torch.cat(pcd1s, dim=0)
#     pcd2s = torch.cat(pcd2s,dim=0)

#     masked_pcds = torch.cat((pcds,pcd1s,pcd2s), dim=0)

#     return masked_pcds

def mask_aug(pcds):
    bs, num, _ = pcds.shape
    device = pcds.device
    pcd1s = []
    pcd2s = []
    pcd3s = []
    pcd4s = []
    for i in range(bs):

        pcd1 = pcds[i]
        # pcd2 = pcds[i]
        # pcd3 = pcds[i]
        # pcd4 = pcds[i]
        [x_min, y_min, z_min], idx = torch.min(pcd1[:, :3], axis=0)
        [x_max, y_max, z_max], idx = torch.max(pcd1[:, :3], axis=0)

        x_center = (x_max + x_min) / 2.
        y_center = (y_max + y_min) / 2.
        z_center = (z_max + z_min) / 2.
        # x_center, y_center, z_center = torch.mean(corners, 0)

        # view = torch.randint(0, 8, (2,))
        # while view[0] == view[1]:
        #     view = torch.randint(0, 8, (2,))
      
        view = torch.randperm(8)[:2]
        
        pcd1 = split(pcd1, view[0], x_center, y_center, z_center)
        #pcd2 = split(pcd2,view[1],x_center,y_center,z_center)
        #pcd3 = split(pcd3,view[2],x_center,y_center,z_center)
        # pcd4 = split(pcd4,view[3],x_center,y_center,z_center)
        pcd1 = resample_pcd(pcd1, num)
        #pcd2 = resample_pcd(pcd2, num)
        #pcd3 = resample_pcd(pcd3, num)
        # pcd4 = resample_pcd(pcd4, num)
        pcd1s.append(pcd1.unsqueeze(0))
        #pcd2s.append(pcd2.unsqueeze(0))
        #pcd3s.append(pcd3.unsqueeze(0))
        # pcd4s.append(pcd4.unsqueeze(0))

    pcd1s = torch.cat(pcd1s, dim=0)
    #pcd2s = torch.cat(pcd2s,dim=0)
    #pcd3s = torch.cat(pcd3s, dim=0)
    # pcd4s = torch.cat(pcd4s,dim=0)

    masked_pcds = torch.cat((pcds,pcd1s), dim=0)

    return masked_pcds

# def mask_aug(pcds):
#     bs, num, _ = pcds.shape
#     device = pcds.device
#     pcd1s = []
#     pcd2s = []
#     pcd3s = []
#     pcd4s = []
#     for i in range(bs):

#         pcd1 = pcds[i]
#         pcd2 = pcds[i]
#         pcd3 = pcds[i]
#         pcd4 = pcds[i]
#         [x_min, y_min, z_min], idx = torch.min(pcd1[:, :3], axis=0)
#         [x_max, y_max, z_max], idx = torch.max(pcd1[:, :3], axis=0)

#         x_center = (x_max + x_min) / 2.
#         y_center = (y_max + y_min) / 2.
#         z_center = (z_max + z_min) / 2.
#         # x_center, y_center, z_center = torch.mean(corners, 0)

#         # view = torch.randint(0, 8, (2,))
#         # while view[0] == view[1]:
#         #     view = torch.randint(0, 8, (2,))
      
#         view = torch.randperm(8)[:3]
        
#         pcd1 = split(pcd1, view[0], x_center, y_center, z_center)
#         pcd2 = split(pcd2,view[1],x_center,y_center,z_center)
#         pcd3 = split(pcd3,view[2],x_center,y_center,z_center)
#         # pcd4 = split(pcd4,view[3],x_center,y_center,z_center)
#         pcd1 = resample_pcd(pcd1, num)
#         pcd2 = resample_pcd(pcd2, num)
#         pcd3 = resample_pcd(pcd3, num)
#         # pcd4 = resample_pcd(pcd4, num)
#         pcd1s.append(pcd1.unsqueeze(0))
#         pcd2s.append(pcd2.unsqueeze(0))
#         pcd3s.append(pcd3.unsqueeze(0))
#         # pcd4s.append(pcd4.unsqueeze(0))

#     pcd1s = torch.cat(pcd1s, dim=0)
#     pcd2s = torch.cat(pcd2s,dim=0)
#     pcd3s = torch.cat(pcd3s, dim=0)
#     # pcd4s = torch.cat(pcd4s,dim=0)

#     masked_pcds = torch.cat((pcds,pcd1s,pcd2s,pcd3s), dim=0)

#     return masked_pcds


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = furthest_point_sample(data, number) 
    fps_data = gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc



def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        # lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        warming_up_t = getattr(config, 'warmingup_e', 0)
        lr_lbmd = lambda e: max(config.lr_decay ** ((e - warming_up_t) / config.decay_step), config.lowest_decay) if e >= warming_up_t else max(e / warming_up_t, 0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, config):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler

    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)



def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        # if isinstance(crop,list):
        INPUT.append(fps(input_data,2048))
        CROP.append(fps(crop_data,2048))
        # else:
        #    INPUT.append(input_data)
        #    CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

def get_ptcloud_img(ptcloud,roll,pitch):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(roll,pitch)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img



def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y', 
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    

def random_scale(partial, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale
