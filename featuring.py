from torch.utils.data import DataLoader
import argparse
import torch
from dataset import Dataset
from fastprogress import progress_bar
from config import update_config
from model.CLIPEncoder import CLIPEncoder
import h5py
import os
from util import str2bool
import numpy as np
import torch.nn.functional as F


def to_numpy(tensor):
    tensor = tensor.to('cpu')
    output = np.array(tensor)
    return output

def save():
    parser = argparse.ArgumentParser(description='Contrastive_learning')
    parser.add_argument('--dataset', default='shanghai-sd', type=str, help='The name of the dataset to train.')
    parser.add_argument('--clip_length', default=1, type=int, help='number of frames')
    parser.add_argument('--save_mode', default='training', type=str, help='train or test')

    args = parser.parse_args()
    cfg = update_config(args, mode='save')
    cfg.print_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    # pretrained CLIP Encoder
    encoder = CLIPEncoder('ViT-L/14').to(device)
    encoder.eval()

    # dataloader
    dataset_name = cfg.dataset
    if cfg.save_mode == 'training':
        video_folders = os.listdir(cfg.train_data)
        video_folders.sort()
        video_folders = [os.path.join(cfg.train_data, aa) for aa in video_folders]
    else:
        video_folders = os.listdir(cfg.test_data)
        video_folders.sort()
        video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    # setting
    save_path = f"features/{dataset_name}/frame/{cfg.save_mode}"
    os.makedirs(save_path, exist_ok=True)

    # featuring
    for i in progress_bar(range(len(video_folders)), total=len(video_folders)):
        folder = video_folders[i]
        dataset = Dataset(cfg, folder)
        folder_name = folder.split('/')[-1]
        save_file_path = f"{save_path}/{folder_name}.h5"
        one_video_features = []

        for c in range(len(dataset)):
            clip = dataset[c]
            clip = clip.unsqueeze(0).to(device) # (1, T, C, H, W)
            cft = encoder(clip).squeeze(0) # (T, D)
            one_video_features.append(to_numpy(cft))

        # save features per video
        with h5py.File(save_file_path, 'w') as f:
            for v, v_ft in enumerate(one_video_features):
                f.create_dataset(str(v).zfill(4), data=v_ft)

        print(f'{i+1}/{len(video_folders)}: {save_file_path} ok!')
            

if __name__=="__main__":
    save()