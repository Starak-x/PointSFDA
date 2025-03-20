import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

#from lightly import loss as losses
import torch.nn.functional as F
import torch.nn as nn
#from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from torch.autograd import Function
import numpy as np
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from extensions.expansion_penalty.expansion_penalty_module import expansionPenaltyModule
chamfer_dist = chamfer_3DDist()
penalty_func = expansionPenaltyModule()

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

def chamfer(p1, p2):  # L2 Chamfer Distance
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):  # L1 Chamfer Distance
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)

    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def get_loss(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)
    
    #loss_all = (cdc + cd1 + cd2 + cd3) * 1e3 

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3 ##############
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def get_AdaPoinTr_loss(ret, gt, sqrt=False):
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side
    pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret  # [16,512,3] [16,64,3] [16,2048,3] [16,16384,3]

    assert pred_fine.size(1) == gt.size(1)
    factor=8
    # denoise loss       factor=32
    idx = knn_point(factor, gt, denoised_coarse)  # B n k [16,64,32]
    denoised_target = index_points(gt, idx)  # B n k 3 [16,64,32,3]
    denoised_target = denoised_target.reshape(gt.size(0), -1, 3)  # [8, 512, 3]

    assert denoised_target.size(1) == denoised_fine.size(1)
    loss_denoised = CD(denoised_fine, denoised_target)
    loss_denoised = loss_denoised * 0.5

    # recon loss
    loss_coarse = CD(pred_coarse, gt)
    loss_fine = CD(pred_fine, gt)
    loss_recon = loss_coarse + loss_fine

    return loss_denoised, loss_recon

def get_SVDFormer_loss(pcds_pred,gt, sqrt=True,alpha1=1,alpha2=1):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2 = pcds_pred

    gt_1 = fps_subsample(gt, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt)
    # partial_matching = PM(partial, P2)


    loss_all = cdc + alpha1*cd1 + alpha2*cd2
    losses = [cdc, cd1, cd2]
    return loss_all, losses

# def get_loss(pcds_preds,partial, gt, sqrt=True):
#     """loss function
#     Args
#         pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
#     """
#     if sqrt:
#         CD = chamfer_sqrt
#         PM = chamfer_single_side_sqrt
#     else:
#         CD = chamfer
#         PM = chamfer_single_side
#
#     coarse,fine = pcds_preds
#
#     gt_2 = fps_subsample(gt, fine.shape[1])
#     gt_1 = fps_subsample(gt,coarse.shape[1])
#
#     cd_coarse = CD(coarse,gt_1)
#     cd_fine = CD(fine,gt_2)
#
#     partial_matching = PM(partial, fine)
#
#     loss_all = (cd_fine+cd_coarse+0.5*partial_matching) * 1e3
#     losses = [cd_coarse,cd_fine,partial_matching]
#     return loss_all, losses

def get_loss_pcn(pcds_pred, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    coarse, fine = pcds_pred

    gt_2 = fps_subsample(gt, fine.shape[1])
    gt_1 = fps_subsample(gt, coarse.shape[1])

    cd_coarse = CD(coarse, gt_1)
    cd_fine = CD(fine, gt_2)

    loss_all = (cd_fine + cd_coarse) * 1e3
    losses = [cd_coarse, cd_fine]
    return loss_all, losses


def hausdorff(partial, pcd_pred):
    """
    :param partial: (B, 3, N)  partial
    :param pcd_pred: (B, 3, M) output
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = partial.shape[2]
    n_pts2 = pcd_pred.shape[2]

    pc1 = partial.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2))  # (B, 3, N, M)
    pc2 = pcd_pred.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1))  # (B, 3, N, M)
    # print(pc1.shape, pc2.shape)##########################################333
    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1))  # (B, N, M)
    shortest_dist, _ = torch.min(l2_dist, dim=2)
    hausdorff_dist, _ = torch.max(shortest_dist, dim=1)  # (B, )
    hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


def get_real_loss(pcd_pred, partial, gt=None, sqrt=True):
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side
    ucd = PM(partial, pcd_pred)
    uhd = hausdorff(partial.permute([0, 2, 1]), pcd_pred.permute([0, 2, 1]))  ##################33
    # uhd = hausdorff(partial.permute([0,2,1]), pcd_pred.permute([0,2,1]))
    # loss = ucd*1e2 + uhd*0.5
    if gt is not None:
        cd = CD(pcd_pred, gt)
        return [uhd, ucd, cd]
    # return loss,[uhd, ucd]
    return [uhd,ucd]


# def get_real_loss(pcd_pred,partial, gt=None, sqrt=True):
#     if sqrt:
#         CD = chamfer_sqrt
#         PM = chamfer_single_side_sqrt
#     else:
#         CD = chamfer
#         PM = chamfer_single_side
#     ucd = PM(partial, pcd_pred)
#     uhd = hausdorff(partial.permute([0,2,1]), pcd_pred.permute([0,2,1]))
#     loss = ucd*1e2 + uhd*0.5
#     if gt is not None:
#         cd = CD(pcd_pred, gt)
#         uhd = hausdorff(partial.permute([0,2,1]), pcd_pred.permute([0,2,1]))##################33
#         return [uhd, ucd, cd]
#     return loss,[uhd, ucd]


def get_cd(coarse_pcd, coarse_source, sqrt=True):
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side
    #print(coarse_source.shape,coarse_pcd.shape)
    coarse_source = fps_subsample(coarse_source, coarse_pcd.shape[1])
    cd = CD(coarse_pcd, coarse_source)
    return cd


def get_ucd(partial, pcd_pred, sqrt=True):
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    ucd = PM(partial, pcd_pred)
    return ucd


# def get_distill_loss(student_feat,teacher_feat,loss_type):#[b,512,1]
#     #print(student_feat.shape, teacher_feat.shape)
#     assert teacher_feat.shape == student_feat.shape
#     batch_size,_,_ = student_feat.shape
#     if loss_type == 'cosine':
#         loss_func = losses.NegativeCosineSimilarity()
#     elif self.loss_type == 'l2':
#         loss_func = nn.MSELoss(reduction='mean')
#     elif loss_type == 'smoothl1':
#         loss_func = nn.SmoothL1Loss(reduction='mean')
#     elif loss_type == 'ntxent':
#         loss_func = losses.NTXentLoss(temperature=0.07)
#     elif loss_type == 'barlow':
#         loss_func = losses.BarlowTwinsLoss(lambda_param=5.e-3)
#
#     loss = torch.FloatTensor([0.0]).cuda()
#     if not 'l1' in loss_type and not 'l2' in loss_type:
#         for batch_id in range(batch_size):
#             if loss_type == 'cosine':
#                 loss += (1 + loss_func(student_feat[batch_id], teacher_feat[batch_id]).mean())
#             #else:
#                 #loss += (loss_func(student_feat[batch_id], teacher_feat[batch_id]) / num_mask)
#
#         loss = loss.mean() / batch_size
#     else:
#         loss = loss_func(student_feat, teacher_feat)
#
#     return loss

def choose_points(source_pcd, pcd_pred, nbr_size=16, num=128):
    _, dist = pointops.knn(source_pcd, pcd_pred, nbr_size)  # [16,256,32]
    dist_sum = torch.sum(dist, dim=-1)
    # dist_max = torch.max(dist_sum,dim=-1)
    # dist_avg = torch.mean(dist_sum,dim=-1)
    idx = torch.argsort(dist_sum, dim=1, descending=True)[:, 0:num]
    b, _ = idx.shape
    coarse_choose = torch.gather(source_pcd, dim=1, index=idx.unsqueeze(-1).expand(b, num, 3))
    return coarse_choose
    # threshold = 12.
    # t= dist_sum <threshold
    # #dist_sum = dist_sum[dist_sum<threshold]
    # idx_choose = source_pcd[dist_sum<threshold,:]


# def get_distill_loss(source_feats,target_feats):
#     bs,_,_ = source_feats[0].shape
#     for i in range(len(source_feats)):
#         source_feats[i] = source_feats[i].view(bs,-1)
#         target_feats[i] = target_feats[i].view(bs, -1)
#         similarity = F.cosine_similarity(source_feats[i],target_feats[i],dim=-1)
#         if i==0:
#             loss = similarity.mean()
#         else:
#             loss += similarity.mean()
#
#     return loss

# def get_distill_loss(source_feats, target_feats):
#     bs, _, _ = source_feats.shape
#     print(len(source_feats))
#     for i in range(len(source_feats)):
#         source_feats[i] = source_feats[i].view(bs, -1)
#         target_feats[i] = target_feats[i].view(bs, -1)
#         similarity = F.cosine_similarity(source_feats[i], target_feats[i], dim=-1)
#         if i == 0:
#             loss = similarity.mean()
#         else:
#             loss += similarity.mean()

#     return loss

def get_distill_loss(source_feats, target_feats):
    bs, _, _ = source_feats.shape
    # print(len(source_feats))
    # for i in range(len(source_feats)):
    source_feats = source_feats.view(bs, -1)
    target_feats = target_feats.view(bs, -1)
   
    similarity = F.cosine_similarity(source_feats, target_feats, dim=-1)

    loss = 1.- similarity.mean()
    return loss


# class ManifoldnessConstraint(nn.Module):
#     """
#     The Normal Consistency Constraint
#     """
#     def __init__(self, support=8, neighborhood_size=32):
#         super().__init__()
#         self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
#         self.support = support
#         self.neighborhood_size = neighborhood_size
#
#     def forward(self, xyz):
#
#         normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)#[32, 2048, 3]
#
#         idx = pointops.knn(xyz, xyz, self.support)[0]#[32, 2048, 16]
#         neighborhood = pointops.index_points(normals, idx)#[32, 2048, 16, 3]
#
#         cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)#[32, 2048, 16]
#         penalty = 1 - cos_similarity
#         penalty = penalty.std(-1) #[32, 2048]
#         penalty = penalty.mean(-1)#[32]
#
#         return penalty
def get_manifold_loss(pcd, support=16, neighborhood_size=32):
    """
    The Normal Consistency Constraint
    """
    normals = estimate_pointcloud_normals(pcd, neighborhood_size=neighborhood_size)  # [32, 2048, 3]

    idx = pointops.knn(pcd, pcd, support)[0]  # [32, 2048, 16]
    neighborhood = pointops.index_points(normals, idx)  # [32, 2048, 16, 3]

    cos_similarity = F.cosine_similarity(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood, dim=3)  # [32, 2048, 16]

    penalty = 1 - cos_similarity  #
    penalty = penalty.std(-1)  # [32, 2048]
    penalty = penalty.mean(-1)  # [32]

    penalty = penalty.mean()
    return penalty


def get_rcd(partial, pred, nbr_size=32, num=128, sqrt=False):
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side
    bs, _, _ = partial.shape
    center = pointops.fps(partial, num)
    idx = pointops.knn(center, pred, nbr_size)[0]
    nbrs_pred = pointops.index_points(pred, idx).reshape(bs, -1, 3)  # [32, 1280, 3]
    cd = CD(nbrs_pred, partial)
    return cd


class NearestDistanceLoss(nn.Module):
    def __init__(self):
        super(NearestDistanceLoss, self).__init__()

    def forward(self, xyz, nbr_size=2, alpha=5.):
        idx, dist = pointops.knn(xyz, xyz, nbr_size)  # [16, 2048, 2]   [16, 2048, 2]
        dist = dist.sum(-1)  # [16, 2048]
        avg_dist = dist.mean(-1, keepdim=True)  # [16]
        print(dist.shape, avg_dist.shape)
        loss = dist[dist > avg_dist * alpha].sum()
        return loss


def get_nearest_nbr_loss(xyz,alpha=1.5, beta=0.75):
    bs,num,_ = xyz.shape

    expand_xyz1 = xyz.unsqueeze(2).expand(-1,-1,num,-1)
    expand_xyz2 = xyz.unsqueeze(1).expand(-1, num, -1, -1)
    dist_matrix = torch.norm(expand_xyz1-expand_xyz2,p=2,dim=3)#[16, 2048, 2048]

    indentity_matrix = torch.eye(num).to(dist_matrix.device)
    dist_matrix = dist_matrix+indentity_matrix
    dist,idxs = dist_matrix.min(dim=2)#[16, 2048] [16, 2048]
    mean_dist = dist.mean(dim=-1).unsqueeze(-1).expand(-1,num)

    loss = torch.sum(torch.clamp(dist-alpha*mean_dist,min=0))

    loss = loss+torch.sum(torch.clamp(beta * mean_dist-dist, min=0))
    return loss
# def get_knearest_nbr_loss(xyz,k=8,alpha=1.5, beta=0.75):
#     idx, _ = pointops.knn(xyz, xyz, k)
#     nbrs = pointops.index_points(xyz, idx)

# def get_knearest_nbr_loss(xyz,k=8,alpha=1.5, beta=0.75):
#     bs,num,_ = xyz.shape
#
#     expand_xyz1 = xyz.unsqueeze(2).expand(-1,-1,num,-1)
#     expand_xyz2 = xyz.unsqueeze(1).expand(-1, num, -1, -1)
#     dist_matrix = torch.norm(expand_xyz1-expand_xyz2,p=2,dim=3)#[16, 2048, 2048]
#
#     indentity_matrix = torch.eye(num).to(dist_matrix.device)
#     dist_matrix = dist_matrix+indentity_matrix
#     k_dist,idx =torch.topk(dist_matrix, k, dim=2, largest=False)
#
#     dist = k_dist.sum(dim=-1)
#     #dist,idxs = dist_matrix.min(dim=2)#[16, 2048] [16, 2048]
#     mean_dist = dist.mean(dim=-1).unsqueeze(-1).expand(-1,num)
#
#     loss = torch.sum(torch.clamp(dist-alpha*mean_dist,min=0))
#
#     #loss = loss+torch.sum(torch.clamp(beta * mean_dist-dist, min=0))
#     return loss

def get_penalty_loss(xyz, primitive_size=64,expansion_alpha=1.2):

    dist, _, mean_mst_dis = penalty_func(xyz, primitive_size, expansion_alpha)
    loss_mst = torch.mean(dist)
    return loss_mst