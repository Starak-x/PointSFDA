

import logging
import os
import torch
import sys
#import utils.data_loaders
import utils.helpers
from datetime import datetime
from tqdm import tqdm
from time import time
import numpy as np
import builder
from tensorboardX import SummaryWriter
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss
from torch.utils.data import DataLoader
from utils.misc import fps_subsample
from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset, RealDataset, GeneratedDataset
from data.ScanSalon_dataset import ScanSalonDataset
#from models.Point_MAE_withoutmask import Point_MAE
from utils.loss_utils import get_loss,get_real_loss,get_distill_loss,get_cd,get_ucd





def validate(cfg, epoch_idx=-1, test_dataset = None, test_data_loader=None, test_writer=None, model=None, source_model=None,test=True):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    cfg.dataset.split = 'test'

    log_path = cfg.train.logs + '.txt'

    if test_data_loader is None:
        if cfg.dataset.name in ['MatterPort','ScanNet','PartNet']:
            test_dataset = RealDataset(cfg)
        elif cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
            test_dataset = GeneratedDataset(cfg)
        elif cfg.dataset.name in ['CRN']:
            test_dataset = CRNShapeNet(cfg)
        elif cfg.dataset.name in['ScanSalon','KITTI']:
            test_dataset = ScanSalonDataset(cfg)

        test_data_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg.test.num_workers,
                pin_memory=True)

    # Setup networks and initialize networks
    if model is None:
        #model = Model(dim_feat=512, up_factors=[2, 2], radius=1)
        #model = Point_MAE()
        model = builder.make_model(cfg)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        print('Recovering from %s ...' % (cfg.test.model_path))
        checkpoint = torch.load(cfg.test.model_path)
        model.load_state_dict(checkpoint['model'])

    if source_model is None:

        source_model = builder.make_model(cfg)
        if torch.cuda.is_available():
            source_model = torch.nn.DataParallel(source_model).cuda()

        print('Recovering from %s ...' % (cfg.test.source_model_path))
        source_checkpoint = torch.load(cfg.test.source_model_path)
        source_model.load_state_dict(source_checkpoint['model'])
    # Switch models to evaluation mode
    model.eval()
    source_model.eval()

    n_samples = len(test_data_loader)
    if cfg.dataset.name in ['KITTI','ScanNet']:
        test_losses = AverageMeter(['ucd_coarse','cd_coarse','uhd', 'ucd'])
    else: 
        test_losses = AverageMeter(['ucd_coarse','cd_coarse','uhd', 'ucd','cdl2'])


    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    categories = test_dataset.cat_ordered_list
    #class_number = test_dataset.class_number
    # Testing loop
    with tqdm(test_data_loader) as t:

        for idx, data in enumerate(t):
            with torch.no_grad():
               
                if cfg.dataset.name in ['ScanNet', 'MatterPort','KITTI']:
                    #'ScanNet', 'MatterPort', 'KITTI' do not have gt, we use ucd for both test and validate
                    partial, label = data
                    partial = partial.float().cuda()
                    pcd_out = model(partial.contiguous())
                    pcd_pred = pcd_out[-1]
                    #coarse_pcd = pcd_out[1]######512
                    coarse_pcd = pcd_out[0]######
                    source_out = source_model(partial)
                    #source_coarse = source_out[1]#####512
                    source_coarse = source_out[0]#####
                    #source_coarse = fps_subsample(source_coarse,128)

                    source_pred = source_out[-1]             
                    sample_source = fps_subsample(source_pred, 256)#
                    loss_coarse = get_ucd(sample_source, pcd_pred, sqrt=False)

                    ucd_coarse = loss_coarse.item() * 1e4

                    
                    loss_cd_coarse = get_cd(coarse_pcd, source_coarse, sqrt=False)
                    cd_coarse = loss_cd_coarse.item() * 1e4


                    losses = get_real_loss(pcd_pred, partial,sqrt=False)  # L2UCD UHD
                    uhd = losses[0].item() * 1e2
                    ucd = losses[1].item() * 1e4
            
                    _metrics = [ucd]
                    test_losses.update([ucd_coarse, cd_coarse, uhd, ucd])
                elif cfg.dataset.name in ['ModelNet', '3D_FUTURE', 'CRN','ScanSalon']:
                    gt, partial, label = data
                    gt = gt.float().cuda()
                    partial = partial.float().cuda()
                    # if cfg.model.name == 'SnowflakeNet':
                    #     feat, pcd_out = model(partial.contiguous())
                    # else:
                    pcd_out = model(partial.contiguous())
                    pcd_pred = pcd_out[-1]
                    coarse_pcd = pcd_out[0]######
                    source_out = source_model(partial)
                    source_coarse = source_out[0]#####


                    source_pred = source_out[-1]             
                    sample_source = fps_subsample(source_pred, 256)#
                    loss_coarse = get_ucd(sample_source, pcd_pred, sqrt=False)
                    ucd_coarse = loss_coarse.item() * 1e4

                    
                    loss_cd_coarse = get_cd(coarse_pcd, source_coarse, sqrt=False)
                    cd_coarse = loss_cd_coarse.item() * 1e4

                    losses = get_real_loss(pcd_pred, partial, gt,sqrt=False)  # L2UCD UHD
                    uhd = losses[0].item() * 1e2
                    ucd = losses[1].item() * 1e4
                    cdl2 = losses[2].item() * 1e4
                    #_metrics = [ucd]########################
                    _metrics = [cdl2]
                    test_losses.update([ucd_coarse, cd_coarse, uhd, ucd,cdl2])
             
                
                test_metrics.update(_metrics)
                
                if cfg.dataset.category not in category_metrics:
                    category_metrics[cfg.dataset.category] = AverageMeter(Metrics.names())
                category_metrics[cfg.dataset.category].update(_metrics)
                t.set_description('Test[%d/%d] Category = %s Sample = %s Losses = %s Metrics = %s' %
                                (idx + 1, n_samples,cfg.dataset.category, idx, ['%.4f' % l for l in test_losses.val()], ['%.4f' % m for m in _metrics]))


    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for category in category_metrics:
        print(category, end='\t')
        print(category_metrics[category].count(0), end='\t')
        for value in category_metrics[category].avg():
            print('%.4f' % value, end='\t')
        print()
    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    with open(log_path, "a") as file_object:
        file_object.write("Testing UCD_Coarse,CD_Coarse,UHD,UCD,CDL2:")
        file_object.write("%.4f" % test_losses.avg(0)+" %.4f" % test_losses.avg(1)+" %.4f" % test_losses.avg(2)+" %.4f" % test_losses.avg(3)+" %.4f" % test_losses.avg(4))
        file_object.write('\n')
    #Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/ucd_coarse', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_coarse', test_losses.avg(1), epoch_idx)
        #test_writer.add_scalar('Loss/Epoch/cdl2', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/uhd', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/ucd', test_losses.avg(3), epoch_idx)
        if cfg.dataset.name in ['ModelNet', '3D_FUTURE', 'CRN']:#################################
            test_writer.add_scalar('Loss/Epoch/cdl2', test_losses.avg(4), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(-1)
