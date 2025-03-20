# -*- coding: utf-8 -*-
# @Author: XP

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





def test(cfg, epoch_idx=-1, test_dataset = None, test_data_loader=None, test_writer=None, model=None, source_model=None,test=True):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    cfg.dataset.split = 'test'

    #log_path = cfg.train.logs + '.txt'

    if test_data_loader is None:
        if cfg.dataset.name in ['MatterPort','ScanNet','KITTI','PartNet']:
            test_dataset = RealDataset(cfg)
        elif cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
            test_dataset = GeneratedDataset(cfg)
        elif cfg.dataset.name in ['CRN']:
            test_dataset = CRNShapeNet(cfg)
        elif cfg.dataset.name in['ScanSalon']:
            test_dataset = ScanSalonDataset(cfg)

        test_data_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg.test.num_workers,
                pin_memory=True)

    # Setup networks and initialize networks
    if model is None:
        model = builder.make_model(cfg)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        print('Recovering from %s ...' % (cfg.test.model_path))
        checkpoint = torch.load(cfg.test.model_path)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()


    n_samples = len(test_data_loader)
 
    test_losses = AverageMeter(['uhd', 'ucd', 'cdl2'])
   
    # if cfg.dataset.name in ['ScanNet', 'MatterPort', 'KITTI']:
    #     test_losses = AverageMeter(['uhd','ucd'])
    # elif cfg.dataset.name in ['ModelNet', '3D_FUTURE', 'CRN']:
    #     test_losses = AverageMeter(['uhd','ucd','cdl2'])

    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    categories = test_dataset.cat_ordered_list
    #class_number = test_dataset.class_number
    # Testing loop
    with tqdm(test_data_loader) as t:

        for idx, data in enumerate(t):
            with torch.no_grad():
        
                
                gt, partial, label = data
                gt = gt.float().cuda()
                partial = partial.float().cuda()
              
                pcd_out = model(partial.contiguous())
                pcd_pred = pcd_out[-1]

                
                
                sample_idx  = torch.randperm(pcd_pred.shape[1])[:256]
                sample_pcd =pcd_pred[:,sample_idx,:].to(pcd_pred.device).contiguous()

                losses = get_real_loss(pcd_pred, partial, gt,sqrt=False)  # L2UCD UHD
                uhd = losses[0].item() * 1e2
                ucd = losses[1].item() * 1e4
                cdl2 = losses[2].item() * 1e4
                #_metrics = [ucd]########################
                _metrics = [cdl2]
                test_losses.update([uhd, ucd, cdl2])

                ######################visualization#############################
                
                output_folder = os.path.join(cfg.train.out_path, cfg.dataset.name, categories[label])  # categories[index[0]]
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                partial_cpu = torch.squeeze(partial).cpu()
                sample_cpu = torch.squeeze(sample_pcd).cpu()
                pred_cpu = torch.squeeze(pcd_pred).cpu()

                partial_np = np.array(partial_cpu)
                pred_np = np.array(pred_cpu)
                sample_np = np.array(sample_cpu)

            
                #np.savetxt(output_folder + '/' + str(idx) + 'pred_wo_sample' + str(int(cdl2)) + '.xyz', pred_np, fmt='%0.8f')

                #np.savetxt(output_folder + '/' + str(idx) + 'partial' + '.xyz', partial_np, fmt='%0.8f')
                np.savetxt(output_folder + '/' + str(idx) + 'baseline' + str(int(cdl2)) + '.xyz', pred_np, fmt='%0.8f')
                #np.savetxt(output_folder + '/' + str(idx) + 'random' + '.xyz', sample_np, fmt='%0.8f')
                # if cfg.dataset.name in ['ModelNet', '3D_FUTURE', 'CRN', 'USSPA', 'ScanSalon']:
                #     gt_cpu = torch.squeeze(gt).cpu()
                #     gt_np = np.array(gt_cpu)
                #     np.savetxt(output_folder + '/' + str(idx) + 'gt' + '.xyz', gt_np, fmt='%0.8f')

                
                test_metrics.update(_metrics)
                if categories[label] not in category_metrics:
                    category_metrics[categories[label]] = AverageMeter(Metrics.names())
                category_metrics[categories[label]].update(_metrics)
                t.set_description('Test[%d/%d] Category = %s Sample = %s Losses = %s Metrics = %s' %
                                (idx + 1, n_samples, categories[label], idx, ['%.4f' % l for l in test_losses.val()], ['%.4f' % m for m in _metrics]))

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
