# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import sys
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm
from time import time
import argparse

from tensorboardX import SummaryWriter
from config_snow import cfg
sys.path.append("..")
from utils import yaml_reader
from datetime import datetime
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss
from SnowflakeNet_model import SnowflakeNet
from torch.utils.data import DataLoader
from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset, RealDataset, GeneratedDataset
import builder

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./SnowflakeNet.yaml', help='Configuration File')
    parser.add_argument('--test', dest='test', help='Test neural networks in source domain', action='store_true')
    args = parser.parse_args()
    return args

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    cfg.dataset.split = 'train'
    if cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
        train_dataset = GeneratedDataset(cfg)
    elif cfg.dataset.name in ['CRN']:
        train_dataset = CRNShapeNet(cfg)
    
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,####
        num_workers=cfg.train.num_workers,
        pin_memory=True)

    cfg.dataset.split = 'test'
    if cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
        val_dataset = GeneratedDataset(cfg)
    elif cfg.dataset.name in ['CRN']:
        val_dataset = CRNShapeNet(cfg)
    
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers//2,
        pin_memory=True)

    output_dir = os.path.join(cfg.train.out_path, datetime.now().isoformat(), '%s')
    cfg.train.model_path = output_dir % 'checkpoints'
    cfg.train.logs = output_dir % 'logs'
    log_path = cfg.train.logs + '.txt'
    if not os.path.exists(cfg.train.model_path):
        os.makedirs(cfg.train.model_path)
    with open(log_path, "a") as file_object:
        file_object.write("Load "+cfg.dataset.name+" dataset")
    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.train.logs, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.train.logs, 'test'))

    model = SnowflakeNet(config=cfg.model)
    #model = PCN()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers

    optimizer, scheduler = builder.build_opti_sche(model, cfg)

    init_epoch = 0
    best_metrics = float('inf')
    steps = 0

    if 'WEIGHTS' in cfg.train:
        logging.info('Recovering from %s ...' % (cfg.train.model_path))
        checkpoint = torch.load(cfg.train.model_path)
        best_metrics = checkpoint['best_metrics']
        model.load_state_dict(checkpoint['model'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.train.epochs + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0
        total_cd_p3 = 0
        total_partial = 0


        batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx,data in enumerate(t):
                #print(len(data))
                gt, partial, index = data
                gt = gt.squeeze(0).cuda()
                partial = partial.squeeze(0).cuda()
                #print(gt.shape,partial.shape,index)
                pcds_preds = model(partial)
               
                loss_total, losses = get_loss(pcds_preds,partial, gt, sqrt=False)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                cd_pc_item = losses[0].item() * 1e4
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e4
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e4
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e4
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e4
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p3', cd_p3_item, n_itr)
                train_writer.add_scalar('Loss/Batch/partial_matching', partial_item, n_itr)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.train.epochs, batch_idx + 1, n_batches))
                #t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_coarse_item, cd_fine_item]])
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item]])
                if cfg.scheduler.type == 'GradualWarmup':
                    if n_itr < cfg.scheduler.kwargs_2.total_epoch:
                        scheduler.step()

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches
        avg_cd3 = total_cd_p3 / n_batches
        avg_partial = total_partial / n_batches
        # avg_coarse = total_coarse / n_batches
        # avg_fine = total_fine / n_batches

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/partial_matching', avg_partial, epoch_idx)

        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.train.epochs, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]))


        # Validate the current model
        cd_eval = test_net(cfg, epoch_idx, val_dataset, val_data_loader, val_writer, model,test=False)

        # Save checkpoints
        if epoch_idx % cfg.train.save_freq == 0 or cd_eval < best_metrics:
            file_name = 'ckpt-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            #file_name = 'ckpt-best.pth'  if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.train.model_path, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': best_metrics,
                'model': model.state_dict()
            }, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                with open(log_path, "a") as file_object:
                    file_object.write('Save ckpt-best.pth...................\n')

    train_writer.close()
    val_writer.close()


def test_net(cfg, epoch_idx=-1, test_dataset=None, test_data_loader=None, test_writer=None, model=None,test=True):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        cfg.dataset.split = 'test'
        if cfg.dataset.name in ['ModelNet', '3D_FUTURE']:
            test_dataset = GeneratedDataset(cfg)
        elif cfg.dataset.name in ['CRN']:
            test_dataset = CRNShapeNet(cfg)
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True)

    # Setup networks and initialize networks
    if model is None:
        model = SnowflakeNet(config=cfg.model)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.test.model_path))
        checkpoint = torch.load(cfg.test.model_path)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    category_list = test_dataset.cat_ordered_list
    if not test:
        log_path = cfg.train.logs + '.txt'

    with tqdm(test_data_loader) as t:
        for idx, data in enumerate(t):
            with torch.no_grad():
                gt, partial, label = data
                gt = gt.cuda()
                partial = partial.cuda()

                pcds_pred = model(partial.contiguous())
                loss_total, losses = get_loss(pcds_pred, partial, gt, sqrt=False)
                coarse = pcds_pred[0]

                pcd_pred = pcds_pred[-1]

                cdc = losses[0].item() * 1e4
                cd1 = losses[1].item() * 1e4
                cd2 = losses[2].item() * 1e4
                cd3 = losses[3].item() * 1e4
                partial_matching = losses[4].item() * 1e4

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])


                test_metrics.update(_metrics)
                if category_list[label] not in category_metrics:
                    category_metrics[category_list[label]] = AverageMeter(Metrics.names())
                category_metrics[category_list[label]].update(_metrics)


                # if test:
                #     output_folder = os.path.join(cfg.train.out_path, cfg.dataset.name,category_list[label])  # categories[class_idx[0]]
                #     if not os.path.exists(output_folder):
                #         os.makedirs(output_folder)
                #     partial_cpu = torch.squeeze(partial).cpu()
                #     pred_cpu = torch.squeeze(pcd_pred).cpu()
                #     partial_np = np.array(partial_cpu)
                #     pred_np = np.array(pred_cpu)
                #     gt_cpu = torch.squeeze(gt).cpu()
                #     gt_np = np.array(gt_cpu)
                #     coarse_cpu = torch.squeeze(coarse).cpu()
                #     coarse_np = np.array(coarse_cpu)
                #
                #
                #     np.savetxt(output_folder + '/' + category_list[label] + '/partial' + '.xyz', partial_np, fmt='%0.8f')
                #     np.savetxt(output_folder + '/' + str(class_idx) + '/pred' + str(int(cd_fine)) + '.xyz', pred_np, fmt='%0.8f')
                #     np.savetxt(output_folder + '/' + str(class_idx) + '/coarse' + '.xyz', coarse_np, fmt='%0.8f')
                #     np.savetxt(output_folder + '/' + str(class_idx) + '/gt' + '.xyz', gt_np, fmt='%0.8f')


                t.set_description('Test[%d/%d] Category = %s Sample = %s Losses = %s Metrics = %s' %
                                  (idx + 1, n_samples, category_list[label], idx,
                                   ['%.4f' % l for l in test_losses.val()], ['%.4f' % m for m in _metrics]))

    # Print testing results

    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()
    if not test:
        with open(log_path, "a") as file_object:
            msg = "##########EPOCH {:0>4d}##########".format(epoch_idx)
            file_object.write(msg + '\n')
            for category in category_metrics:
                file_object.write(category + '\t')
                file_object.write(str(category_metrics[category].count(0)) + '\t')
                for value in category_metrics[category].avg():
                    file_object.write('%.4f' % value + '\t')
                file_object.write('\n')
            file_object.write('Overall\t\t\t')
            for value in test_metrics.avg():
                file_object.write('%.4f' % value + '\t')
            file_object.write('\n')

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

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cdc', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd1', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd2', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd3', test_losses.avg(3), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/partial_matching', test_losses.avg(4), epoch_idx)


        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(3)

if __name__ == "__main__":
    
    args = get_args_from_command_line()

    cfg = yaml_reader.read_yaml(args.config)
   
    set_seed(cfg.train.seed)
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    #args = get_args_from_command_line()
    print('cuda available ', torch.cuda.is_available())

    # Print config
    print('Use config:')
    pprint(cfg)

    if args.test:
        test_net(cfg)
    else:
        train_net(cfg)