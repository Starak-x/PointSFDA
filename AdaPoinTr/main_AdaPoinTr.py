# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
import argparse
import sys
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from AdaPoinTr_model import AdaPoinTr

sys.path.append("..")
import builder
from utils import yaml_reader
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.loss_utils import *
from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset, RealDataset, GeneratedDataset
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./AdaPoinTr.yaml', help='Configuration File')
    parser.add_argument('--test', dest='test', help='Test neural networks in source domain', action='store_true')
    args = parser.parse_args()
    return args

def train(cfg):
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
        shuffle=True,
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
        num_workers=cfg.test.num_workers,
        pin_memory=True)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.train.out_path, datetime.now().isoformat(), '%s', )
    cfg.train.checkpoints = output_dir % 'checkpoints'
    cfg.train.logs = output_dir % 'logs'
    log_path = cfg.train.logs + '.txt'

    if not os.path.exists(cfg.train.checkpoints):
        os.makedirs(cfg.train.checkpoints)
    with open(log_path, "a") as file_object:
        file_object.write("Load "+cfg.dataset.name+" dataset\n")
    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.train.logs, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.train.logs, 'test'))

    model = AdaPoinTr(config=cfg.model)
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

        total_denoised = 0
        total_recon = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        with tqdm(train_data_loader) as t:
            for batch_idx,data in enumerate(t):
                #print(len(data))
                gt, partial, index = data
                gt = gt.cuda()
                partial = partial.cuda()
                #print(partial.shape,gt.shape)
                #print(gt.shape,partial.shape,index)
                ret = model(partial)
              
                loss_denoised, loss_recon = model.module.get_loss(ret, gt)
                loss_total = loss_recon+loss_denoised

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                cd_denoised = loss_denoised.item() * 1e4
                total_denoised += cd_denoised
                cd_recon = loss_recon.item() * 1e4
                total_recon += cd_recon


                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_denoised', cd_denoised, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_recon', cd_recon, n_itr)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.train.epochs, batch_idx + 1, n_batches))

                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_denoised,cd_recon]])
                if cfg.scheduler.type == 'GradualWarmup':
                    if n_itr < cfg.scheduler.kwargs_2.total_epoch:
                        scheduler.step()


        avg_denoised = total_denoised / n_batches
        avg_recon = total_recon  / n_batches
        #lr_scheduler.step()
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/avg_denoised', avg_denoised, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/avg_recon', avg_recon, epoch_idx)

        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.train.epochs, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_denoised,avg_recon]]))


        # Validate the current model
        cd_eval = test(cfg, epoch_idx, val_dataset, val_data_loader, val_writer, model,test=False)

        # Save checkpoints
        # if epoch_idx % cfg.train.save_freq == 0 or cd_eval < best_metrics:
        #     file_name = 'ckpt-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
        #     output_path = os.path.join(cfg.train.out_path, file_name)
        #     torch.save({
        #         'epoch_index': epoch_idx,
        #         'best_metrics': best_metrics,
        #         'model': model.state_dict()
        #     }, output_path)
        #     logging.info('Saved checkpoint to %s ...' % output_path)
        #     if cd_eval < best_metrics:
        #         best_metrics = cd_eval

        if epoch_idx % cfg.train.save_freq == 0 or cd_eval < best_metrics:
            file_name = 'ckpt-best-%03d.pth'% epoch_idx if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            #file_name = 'ckpt-best.pth' if cd_eval < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.train.checkpoints, file_name)
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
def test(cfg, epoch_idx=-1, test_dataset = None, test_data_loader=None, test_writer=None, model=None,test=True):
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
            num_workers=cfg.test.num_workers,
            pin_memory=True)

    # Setup networks and initialnetworks
    if model is None:
        model = AdaPoinTr(config=cfg.model)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        logging.info('Recovering from %s ...' % (cfg.test.model_path))
        checkpoint = torch.load(cfg.test.model_path)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    if not test:
        log_path = cfg.train.logs + '.txt'
    test_losses = AverageMeter(['cd_coarse','cd_fine'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    categories = test_dataset.cat_ordered_list

    # Testing loop
    with tqdm(test_data_loader) as t:
        for idx, data in enumerate(t):
            with torch.no_grad():
                gt, partial, label = data
                gt = gt.cuda()
                partial = partial.cuda()
                #b, n, _ = partial.shape

                ret = model(partial.contiguous())
                coarse = ret[0]
                fine = ret[-1]
                #loss_total, losses = get_loss_pcn(pcds_pred, gt, sqrt=False)
                loss_coarse = get_cd(coarse,gt,sqrt=False)
                loss_fine = get_cd(fine, gt, sqrt=False)



                cd_fine = loss_fine.item() * 1e4
                cd_coarse = loss_coarse.item()*1e4

                _metrics = [cd_fine]
                test_losses.update([cd_coarse,cd_fine])


                ######################visualization#############################
                # if test:
                #     output_folder = os.path.join(cfg.train.out_path, cfg.dataset.name, cfg.dataset.category)
                #     if not os.path.exists(output_folder):
                #         os.makedirs(output_folder)
                #     partial_cpu = torch.squeeze(partial).cpu()
                #     pred_cpu = torch.squeeze(fine).cpu()
                #     partial_np = np.array(partial_cpu)
                #     pred_np = np.array(pred_cpu)
                #     np.savetxt(output_folder + '/' + str(idx) + 'partial' + '.xyz', partial_np, fmt='%0.8f')
                #     np.savetxt(output_folder + '/' + str(idx) + 'pred' + '.xyz', pred_np, fmt='%0.8f')
                #     if cfg.dataset.name in ['ModelNet', '3D_FUTURE', 'CRN']:
                #         gt_cpu = torch.squeeze(gt).cpu()
                #         gt_np = np.array(gt_cpu)
                #         np.savetxt(output_folder + '/' + str(idx) + 'gt' + '.xyz', gt_np, fmt='%0.8f')

                test_metrics.update(_metrics)
                if categories[label] not in category_metrics:
                    category_metrics[categories[label]] = AverageMeter(Metrics.names())
                category_metrics[categories[label]].update(_metrics)
                
                t.set_description('Test[%d/%d] Category = %s Sample = %s Losses = %s Metrics = %s' %
                                  (idx + 1, n_samples, categories[label], idx, ['%.4f' % l for l in test_losses.val()
                                                                                ], ['%.4f' % m for m in _metrics]))


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


        test_writer.add_scalar('Loss/Epoch/cd_coarse', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/cd_fine', test_losses.avg(1), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(-1)

if __name__ == "__main__":

    args = get_args_from_command_line()

    cfg = yaml_reader.read_yaml(args.config)

    set_seed(cfg.train.seed)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.train.gpu)
    if args.test:
        test(cfg)
    else:
        train(cfg)
