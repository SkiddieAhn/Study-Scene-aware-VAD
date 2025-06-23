import glob
import torch
import numpy as np
import cv2
import torch.nn.functional as F


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def denormalize_image(normalized_img):
    """
    - input: normalized_img (C, H, W), float32,[-1.0, 1.0]
    - output: denormalized_img (H, W, C), uint8,  [0, 255]
    """
    img = np.transpose(normalized_img, [1, 2, 0])  # (C, H, W) → (H, W, C)
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)  # -1~1 → 0~255
    return img


def convert_clip_labels(labels, clip_length):
    clip_labels = []
    total_frames = len(labels)
    num_clips = total_frames // clip_length  

    for i in range(num_clips):
        clip = labels[i * clip_length:(i + 1) * clip_length]
        clip_label = 1 if any(clip) else 0
        clip_labels.append(clip_label)

    return clip_labels


def z_score(arr, eps=1e-8):
    mean = np.mean(arr)
    std_dev = np.std(arr) + eps  # Avoid division by zero
    z_scores = (arr - mean) / std_dev
    return z_scores


def mse_error(pred, target):
    # [1, T, D] → [T, D] 
    if pred.dim() == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    if target.dim() == 3 and target.shape[0] == 1:
        target = target.squeeze(0)
    mse_per_timestep = F.mse_loss(pred, target, reduction='none').mean(dim=1)
    return mse_per_timestep.max()


# def mse_error(pred, target) :
#     # [1, T, D] → [T, D]
#     if pred.dim() == 3 and pred.shape[0] == 1:
#         pred = pred.squeeze(0)
#     if target.dim() == 3 and target.shape[0] == 1:
#         target = target.squeeze(0)
#     return F.mse_loss(pred, target, reduction='mean')


def min_max_normalize(arr, eps=1e-8):
    min_val = np.min(arr)
    max_val = np.max(arr)
    denominator = max_val - min_val + eps  # Avoid division by zero
    normalized_arr = (arr - min_val) / denominator
    return normalized_arr


def create_supcon_labels(scene_length, positive_video_length):
    labels = torch.arange(scene_length).repeat_interleave(positive_video_length)
    return labels


def scene_labeler(data_path):
    paths = sorted(glob.glob(f'{data_path}/*'))
    scene_labels = []
    start = 0 
    for i, path in enumerate(paths):
        name = path.split('/')[-1].split('_')[0]
        if i!=0 and crt_name != name:
            start += 1
        scene_labels.append(start)
        crt_name = name
    return scene_labels


def scene_spliter(data_path):
    paths = sorted(glob.glob(f'{data_path}/*'))
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
