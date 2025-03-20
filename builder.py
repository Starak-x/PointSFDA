import sys
import torch
#from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from utils.schedular import GradualWarmupScheduler

from SnowflakeNet.SnowflakeNet_model import SnowflakeNet
from AdaPoinTr.AdaPoinTr_model import AdaPoinTr
from PCN.PCN_model import PCN
from utils.misc import build_lambda_sche, build_lambda_bnsche
from SeedFormer.SeedFormer_model import SeedFormer



def make_model(cfg):
     
    if cfg.model.name == 'SnowflakeNet':
        #model = SnowflakeNet(dim_feat=512,up_factors=[2,2])
        model = SnowflakeNet(cfg.model)
    elif cfg.model.name == 'AdaPoinTr':
        model = AdaPoinTr(config=cfg.model)
    elif cfg.model.name == 'PCN':
        model = PCN(config=cfg.model)
    elif cfg.model.name == 'SeedFormer':
        model = SeedFormer(config=cfg.model)
    else:
        raise NotImplementedError()
 
    return model




def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sche_config.kwargs.decay_step, gamma=sche_config.kwargs.gamma)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sche_config.kwargs_1.decay_step, gamma=sche_config.kwargs_1.gamma)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr,multiplier=1, total_epoch=sche_config.kwargs_2.total_epoch) #**sche_config.kwargs_2
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=sche_config.kwargs.t_max,
                                      lr_min=sche_config.kwargs.min_lr,
                                      warmup_t=sche_config.kwargs.initial_epochs,
                                      t_in_epochs=True)
    else:
        raise NotImplementedError()

    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]

    return optimizer, scheduler