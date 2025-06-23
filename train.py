from torch.utils.data import DataLoader
import argparse
from dataset import TrainDatasetF
from config import update_config
from util import str2bool
from train_folder.train_pre_func import * 
from train_folder.train_func import Trainer


def main():
    parser = argparse.ArgumentParser(description='Contrastive_learning')
    parser.add_argument('--wandb', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--dataset', default='shanghai-sd', type=str, help='The name of the dataset to train.')
    parser.add_argument('--iters', default=5000, type=int, help='The total iteration number.')
    parser.add_argument('--resume', default=None, type=str, help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
    parser.add_argument('--val_interval', default=100, type=int)
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
    parser.add_argument('--scene_length', default=4, type=int, help='number of scenes in dataset')
    parser.add_argument('--video_length', default=30, type=int, help='number of positive video segments(clips)')
    parser.add_argument('--clip_length', default=16, type=int, help='number of frames in segment(clip)')
    parser.add_argument('--training_mode', default=1, type=int, help='0: recon, 1: recon+cnt, 2: recon+cnt+ce')

    args = parser.parse_args()
    train_cfg = update_config(args, mode='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Pre-work for Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # init wandb
    if train_cfg.wandb:
        init_wandb(train_cfg)
    
    # setup seed (for deterministic behavior)
    seed(seed_value=train_cfg.manualseed)

    # get dataset and loader
    train_dataset = TrainDatasetF(train_cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size)
    print_infor(cfg=train_cfg, dataloader=train_dataloader)

    # define model
    model = def_model(train_cfg, device)

    # define losses
    losses = def_losses(train_cfg, device)

    # define optimizer 
    optim = def_optim(train_cfg, model)

    # load model, optim
    load_model_optim(train_cfg, model, optim)

    # load iter, epoch
    iteration, epoch = load_iter_epoch(train_cfg, train_dataloader)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # train
    Trainer(train_cfg, train_dataset, train_dataloader, model, losses, optim, iteration, epoch, device)


if __name__=="__main__":
    main()