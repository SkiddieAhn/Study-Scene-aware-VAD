import os
import glob
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import h5py


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # to -1 ~ 1
    image_resized = np.transpose(image_resized, [2, 0, 1])  # to (C, W, H)
    return image_resized


class TrainDatasetF():
    def __init__(self, cfg):
        self.data_path = f'features/{cfg.dataset}/frame/training'
        self.scene_indice_list = self.scene_spliter()
        self.scene_total = len(self.scene_indice_list)
        self.videos = []
        self.all_seqs = []
        self.clip_length = cfg.clip_length
        self.video_length = cfg.video_length

        for file in sorted(glob.glob(f'{self.data_path}/*')):
            with h5py.File(file, 'r') as v_features:
                self.videos.append(file)
                all_fts = sorted(list(v_features.keys()))
                random_seq = list(range(len(all_fts)-(self.clip_length-1)))
                random.shuffle(random_seq)
                self.all_seqs.append(random_seq)

    def scene_spliter(self):
        paths = sorted(glob.glob(f'{self.data_path}/*'))
        scene_indice_list = []
        head_idx = 0
        for i, path in enumerate(paths):
            name = path.split('/')[-1].split('_')[0]
            if i == 0:
                crt_name = name
            elif crt_name != name:
                crt_name = name
                scene_indice_list.append((head_idx, i))
                head_idx = i
            elif i == len(paths)-1:
                scene_indice_list.append((head_idx, i+1))
        return scene_indice_list

    def __len__(self):  # This decide the indice range of the PyTorch Dataloader.
        return self.scene_total

    def __getitem__(self, batch_idx):  
        if batch_idx >= len(self):  
            raise IndexError(f"Invalid idx {batch_idx}, max={len(self)-1}")

        scene_range = [self.scene_indice_list[batch_idx][0], self.scene_indice_list[batch_idx][1]-1]
        indice = [random.randint(scene_range[0], scene_range[1]) for _ in range(self.video_length)]
        videos = [self.videos[idx] for idx in indice]
        start_arr = [self.all_seqs[idx][-1] for idx in indice]

        all_video_clip = []
        for v, video in enumerate(videos):
            video_clip = []
            start = start_arr[v]
            with h5py.File(video, 'r') as v_features:
                for k in range(start, start + self.clip_length):
                    key = str(k).zfill(4)
                    v_ft = v_features[key][:]
                    video_clip.append(torch.from_numpy(v_ft))
            video_clip = torch.cat(video_clip, dim=0) # [clip_length, D]
            all_video_clip.append(video_clip)
        all_video_clip = torch.stack(all_video_clip) # [video_length, clip_length, D]
        return indice, all_video_clip


class DatasetF:
    def __init__(self, cfg, video_folder, mode='testing'):
        self.clip_length = cfg.clip_length
        folder_name = video_folder.split('/')[-1]
        self.video = f'features/{cfg.dataset}/frame/{mode}/{folder_name}.h5'
        with h5py.File(self.video, 'r') as v_features:
            self.imgs = sorted(list(v_features.keys()))

    def __len__(self):
        return len(self.imgs) // self.clip_length 

    def __getitem__(self, idx):
        if idx >= len(self):  
            raise IndexError(f"Invalid idx {idx}, max={len(self)-1}")

        video_clip = []
        idx = idx * self.clip_length
        with h5py.File(self.video, 'r') as v_features:
            for k in range(idx, idx + self.clip_length):
                key = str(k).zfill(4)
                v_ft = v_features[key][:]
                video_clip.append(torch.from_numpy(v_ft))
        video_clip = torch.cat(video_clip, dim=0) # [clip_length, D]
        return video_clip


class Dataset:
    def __init__(self, cfg, video_folder):
        self.img_h, self.img_w = cfg.img_size[0], cfg.img_size[1]
        self.clip_length = cfg.clip_length
        self.imgs = glob.glob(video_folder + '/*.jpg')
        self.imgs = self.file_sort(self.imgs)

    def file_sort(self, path_list):
        path_dict = {int(path.split('/')[-1].split('.')[0]) : path for path in path_list}
        path_dict = dict(sorted(path_dict.items()))
        new_list = list(path_dict.values())
        return new_list

    def __len__(self):
        return len(self.imgs) // self.clip_length 

    def __getitem__(self, idx):
        if idx >= len(self):  
            raise IndexError(f"Invalid idx {idx}, max={len(self)-1}")

        video_clip = []
        idx = idx * self.clip_length
        for frame_id in range(idx, idx + self.clip_length):
            tensor = np_load_frame(self.imgs[frame_id], self.img_h, self.img_w)
            tensor = tensor.reshape((-1, self.img_h, self.img_w))
            tensor = torch.from_numpy(tensor)
            video_clip.append(tensor)
        video_clip = torch.stack(video_clip)
        return video_clip


class Label_loader:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        gt = self.load_shanghai_sd()
        return gt


    def load_shanghai_sd(self):
        np_list = glob.glob(f'{self.cfg.data_root}/shanghai-sd/testframemask/*_*.npy')
        np_list.sort()

        gt = []
        for npy in np_list:
            gt.append(np.load(npy))

        return gt