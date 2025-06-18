import torch 
import argparse
import numpy as np
from train_folder.losses import *
from model.CLIPEncoder import CLIPEncoder
from model.supconAE import SubconAE
import os
import wandb
import random


def init_wandb(cfg):
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    project_name = cfg.dataset + str(random.randint(1,1000))
    wandb.init(project=project_name)
    

def seed(seed_value):
    if seed_value == -1:
        return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


def print_infor(cfg, dataloader):
    cfg.epoch_size = cfg.iters // len(dataloader)
    cfg.print_cfg() 

    print('\n===========================================================')
    print('Dataloader Ok!')
    print('-----------------------------------------------------------')
    print('[Data Size]:',len(dataloader.dataset))
    print('[Batch Size]:',cfg.batch_size)
    print('[One epoch]:',len(dataloader.dataset)//cfg.batch_size,'step   # (Data Size / Batch Size)')
    print('[Epoch & Iteration]:',cfg.epoch_size,'epoch &', cfg.iters,'step')
    print('-----------------------------------------------------------')
    print('===========================================================')


def def_model(cfg, device):
    model = SubconAE(input_dim=cfg.embed_dim, hidden_dim=256, latent_dim=128, num_classes=cfg.scene_length).to(device)
    model.train()
    return model


def def_losses(cfg, device):
    cnt_loss = SupConLoss().to(device)
    ce_loss = CrossEntropyLoss().to(device)
    mse_loss = MSELoss().to(device)
    tp_loss = TemporalLoss().to(device)
    losses = [cnt_loss, ce_loss, mse_loss, tp_loss]
    return losses


def def_optim(cfg, model):
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    return optim


def load_model_optim(cfg, model, optim):
    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume)['model'])
        optim.load_state_dict(torch.load(cfg.resume)['optim'])


def load_iter_epoch(cfg, dataloader):
    if cfg.resume:
        iter = torch.load(cfg.resume)['iter']
        epoch = int(iter/len(dataloader)) 
    else:
        iter = 0
        epoch = 0
    return iter, epoch
