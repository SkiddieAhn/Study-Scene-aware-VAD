import torch 
import random
from eval import val
from util import create_supcon_labels
from train_folder.train_ing_func import *
from train_folder.show_gpu import showGPU
import time
import datetime
import math
import wandb


class Trainer:
    def __init__(self, cfg, dataset, dataloader, model, losses, opt, iter, epoch, device):
        self.dataset = dataset
        self.dataloader = dataloader
        self.model = model
        self.cnt_loss, self.ce_loss, self.mse_loss, self.tp_loss = losses
        self.opt = opt
        self.iters = cfg.iters
        self.device = device
        self.wandb = cfg.wandb

        # setting label
        self.scene_labels = create_supcon_labels(cfg.scene_length, cfg.video_length).to(device)

        # train init
        self.start_iter = iter
        self.iter_num = iter
        self.epoch_num = epoch
        self.auc = 0
        self.best_auc = 0
        self.Training = True  

        # training start
        self.fit(cfg)


    def update_dataset(self, indice, clip_length):
        for index in indice:
            self.dataset.all_seqs[index].pop()
            length = len(self.dataset.all_seqs[index])
            if length == 0:
                self.dataset.all_seqs[index] = list(range(len(self.dataset.videos[index])-(clip_length-1)))
                random.shuffle(self.dataset.all_seqs[index])


    def save_model(self, cfg):
        model_dict = make_models_dict(self.model, self.opt, self.iter_num+1)
        save_path = f"{cfg.weight_path}/model_{cfg.dataset}_{cfg.training_mode}_{cfg.manualseed}_{self.iter_num+1}.pth"
        torch.save(model_dict, save_path)
        print(f"\nAlready saved: \'{save_path}'.")


    def one_time(self, iter_num, temp, start_iter):
            torch.cuda.synchronize()
            time_end = time.time()
            if iter_num > start_iter:  
                iter_t = time_end - temp
            else:
                iter_t = None
            temp = time_end
            return iter_t, temp


    def one_print(self, loss, iter_t, eta):
        print(f"\n[{self.iter_num + 1}/{self.iters}] loss: {loss:.5f} | auc: {self.auc:.3f} | best_auc: {self.best_auc:.3f} | iter_t: {iter_t:.3f} | remain_t: {eta}")
        if self.wandb:
            wandb.log({"loss": loss.item()})
            wandb.log({"auc": self.auc})


    def one_forward(self, mode, input):
        recon, logits, sfeature = self.model(input) # forward
        cnt_lv = self.cnt_loss(sfeature, self.scene_labels)
        ce_lv = self.ce_loss(logits, self.scene_labels)
        mse_lv = self.mse_loss(recon, input)
        tp_lv = self.tp_loss(recon, input)
        recon_lv = mse_lv + 0.1*tp_lv

        if mode == 0: 
            total_lv = 2.0*recon_lv  
        elif mode == 1: 
            total_lv = 2.0*recon_lv + 1.0*cnt_lv 
        elif mode == 2:
            total_lv = 2.0*recon_lv + 1.0*cnt_lv + 0.1*ce_lv 
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return total_lv


    def fit(self, cfg):
        print('\n===========================================================')
        print('Training Start!')
        print('===========================================================')

        # prepare
        time_temp = 0
        total_loss = 0

        # show GPU
        showGPU()

        while self.Training:
            # one epoch
            for indice, x in self.dataloader:
                indice = torch.flatten(torch.cat(indice), -1) #[used video indice]
                x = x.view(-1, cfg.clip_length, cfg.embed_dim).to(self.device)  # [scene*pos, T, D]

                # update dataset
                self.update_dataset(indice, cfg.clip_length)

                # calculate time
                if self.iter_num > 0:
                    try:
                        iter_t, time_temp = self.one_time(self.iter_num, time_temp, self.start_iter)
                        time_remain = (self.iters - self.iter_num) * iter_t
                        eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    except:
                        eta = '?'

                # forward
                loss = self.one_forward(cfg.training_mode, x)
                total_loss += loss
                if (self.iter_num + 1) % 20 == 0:
                    self.one_print(total_loss / (self.iter_num-self.start_iter), iter_t, eta)

                # optimization
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # val model
                if (self.iter_num+1) % cfg.val_interval == 0:
                    self.auc = val(cfg, self.model, self.iter_num, self.device)
                    self.best_auc = update_best_model(cfg, self.iter_num, self.auc, self.best_auc, self.model, self.opt)

                # save model
                if (self.iter_num+1) % cfg.save_interval == 0:
                    self.save_model(cfg)
                    
                # training end
                if (self.iter_num+1) == self.iters:
                    self.Training = False
                    break
                
                # update iter_num
                self.iter_num += 1

            # update epoch_num
            self.epoch_num += 1

