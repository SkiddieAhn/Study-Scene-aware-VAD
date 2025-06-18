import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import os
import warnings
from dataset import DatasetF, Label_loader
from util import scene_labeler, mse_error, min_max_normalize
from scipy.ndimage import gaussian_filter1d
from eval_folder.save_func import save_auc_result, plot_tsne_eval
from sklearn import metrics

warnings.filterwarnings('ignore')


def val_train_eval(cfg, model, iter, device):
    # dataset 
    dataset_name = cfg.dataset
    test_video_folders = os.listdir(cfg.test_data)
    test_video_folders.sort()
    test_video_folders = [os.path.join(cfg.test_data, aa) for aa in test_video_folders]

    # setting
    save_dir = f"{cfg.result_path}/{dataset_name}/val/{cfg.training_mode}"
    os.makedirs(save_dir, exist_ok=True)

    # inference 
    with torch.no_grad():
        mse_group = []
        scene_feature_list = [[] for _ in range(cfg.scene_length)]

        for i, folder in enumerate(test_video_folders):
            test_dataset = DatasetF(cfg, folder)
            mse_list = []
            for clip in test_dataset:
                clip = clip.unsqueeze(0).to(device)  # [1, T, D]
                recon, _, _ = model(clip)     
                mse = mse_error(recon, clip).detach().cpu()
                mse_list.extend([mse] * cfg.clip_length)
            mse_group.append(np.array(mse_list))

    # anomaly scoring
    gt_loader = Label_loader(cfg)
    gt_arr = gt_loader()
    preds, label_arr = [], []

    for i in range(len(mse_group)):
        predicted = mse_group[i]
        preds.append(predicted)
        label_arr.append(gt_arr[i][:len(predicted)])
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(label_arr, axis=0)

    # best auc with Gaussian smoothing
    best_auc = 0 
    for sigma in range(0, 20):
        if sigma > 0:
            g_preds = gaussian_filter1d(preds, sigma=sigma)
            nm_preds = min_max_normalize(g_preds)
        else:
            nm_preds = preds 
        fpr, tpr, _ = metrics.roc_curve(labels, nm_preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc > best_auc:
            best_preds = nm_preds
            best_auc = auc
    save_auc_result(cfg, save_dir, dataset_name, cfg.training_mode, iter, best_auc)
    return best_auc
