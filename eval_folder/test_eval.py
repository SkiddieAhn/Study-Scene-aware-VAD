from matplotlib import pyplot as plt
from sklearn import metrics
from fastprogress import progress_bar
import numpy as np
import torch
import torch.nn as nn
from dataset import DatasetF, Label_loader
import os
import warnings
from scipy.ndimage import gaussian_filter1d
from eval_folder.save_func import save_score_auc_graph, plot_tsne_eval
from util import mse_error, min_max_normalize, convert_clip_labels, scene_labeler
from torchvision.utils import save_image
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')


def val_test_eval(cfg, model, iter, device):
    # dataset 
    dataset_name = cfg.dataset
    test_video_folders = os.listdir(cfg.test_data)
    test_video_length = len(test_video_folders)
    test_video_folders.sort()
    test_video_folders = [os.path.join(cfg.test_data, aa) for aa in test_video_folders]
    test_scene_labels = scene_labeler(cfg.test_data)

    # setting
    save_dir = f"{cfg.result_path}/{dataset_name}/test/{cfg.trained_model}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/ascores", exist_ok=True)

    # inference 
    with torch.no_grad():
        mse_group = []
        for i, folder in progress_bar(enumerate(test_video_folders), total=test_video_length):
            test_dataset = DatasetF(cfg, folder)
            mse_list = []
            for j, clip in enumerate(test_dataset):
                clip = clip.unsqueeze(0).to(device)  # [1, T, D]
                recon, _, _ = model(clip) # sft: [1, D]
                mse = mse_error(recon, clip).detach().cpu()
                mse_list.extend([mse] * cfg.clip_length)
            mse_group.append(np.array(mse_list))

    # anomaly scoring
    gt_loader = Label_loader(cfg)
    gt_arr = gt_loader()
    preds = []
    label_arr = []

    for i in range(len(mse_group)):
        predicted = mse_group[i]
        preds.append(predicted)
        label = gt_arr[i][:len(predicted)]
        label_arr.append(label)
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(label_arr, axis=0) 

    # calc auc
    best_auc = 0 
    for sigma in range(0, 20):
        if sigma > 0:
            g_preds = gaussian_filter1d(preds, sigma=sigma)
            nm_preds = min_max_normalize(g_preds)
        else:
            nm_preds = min_max_normalize(preds) 
        fpr, tpr, _ = metrics.roc_curve(labels, nm_preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc > best_auc:
            best_preds = nm_preds
            best_auc = auc
    print('* total auc:', best_auc)

    if cfg.visualization:
        # view total score
        anomalies_idx = [i for i,l in enumerate(labels) if l==1] 
        save_score_auc_graph(y='Anomaly Score', answers_idx=anomalies_idx, scores=best_preds, auc=best_auc, file_path=f'{save_dir}/ascores/all_anomaly_score.jpg')

        # view score per video
        for i in progress_bar(range(len(gt_arr)), total=len(gt_arr)):
            video_gt = label_arr[i]
            video_name = test_video_folders[i].split('/')[-1]

            if i == 0:
                len_past = 0
            else:
                len_past = len_past+len_present
            len_present = len(label_arr[i])

            video_pd = best_preds[len_past:len_past+len_present]
            fpr, tpr, _ = metrics.roc_curve(video_gt, video_pd, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            anomalies_idx = [i for i,l in enumerate(video_gt) if l==1] 
            save_score_auc_graph(y='Anomaly Score', answers_idx=anomalies_idx, scores=video_pd, auc=auc, file_path=f'{save_dir}/ascores/{video_name}_anomaly_score.jpg')
        print('auc calulation per video ok!')

        # inference (train dataset)
        train_video_folders = os.listdir(cfg.train_data)
        train_video_folders.sort()
        train_video_folders = [os.path.join(cfg.train_data, aa) for aa in train_video_folders]
        train_scene_labels = scene_labeler(cfg.train_data)

        with torch.no_grad():
            scene_feature_list = [[] for _ in range(cfg.scene_length)]
            for i, folder in enumerate(train_video_folders):
                train_dataset = DatasetF(cfg, folder, 'training')
                scene_label = train_scene_labels[i]
                for clip in train_dataset:
                    clip = clip.unsqueeze(0).to(device)  # [1, T, D]
                    _, _, sft = model(clip)    # sft: [1, D]
                    feature = sft[0].detach().cpu().numpy()
                    scene_feature_list[scene_label].append(feature)

        # t-sne
        tsne_model = TSNE(n_components=2, perplexity=30, n_iter=2000, random_state=0)
        plot_tsne_eval(
            cfg=cfg,
            tsne_model=tsne_model,
            dataset_name=dataset_name,
            scene_feature_list=scene_feature_list,
            save_dir=save_dir
        )
        print('t-sne visualization ok!')
