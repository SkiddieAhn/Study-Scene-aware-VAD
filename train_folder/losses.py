import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import torch.nn.functional as F


class RegularityLoss(nn.Module):
    def __init__(self):
        super(RegularityLoss, self).__init__()

    def forward(self, mu, logvar):
        # mu, logvar: [B, D]
        # KL divergence between N(mu, sigma^2) and N(0, I)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]
        return kl_div.mean()
        

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        return F.cross_entropy(logits, targets)


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_seq, target_seq):
        # pred_seq, target_seq: [B, T, D]
        return F.mse_loss(pred_seq, target_seq, reduction=self.reduction)


class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()

    def forward(self, pred_seq, target_seq):
        # [B, T, D] → [B, T-1, D]
        diff_pred = pred_seq[:, 1:] - pred_seq[:, :-1]
        diff_target = target_seq[:, 1:] - target_seq[:, :-1]

        # L2 distance between frame differences → [B, T-1]
        diff_loss = F.mse_loss(diff_pred, diff_target, reduction='none')  # [B, T-1, D]
        diff_loss = diff_loss.mean(dim=[1, 2])  # mean over T-1 and D

        return diff_loss.mean()  # final scalar



class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B = features.shape[0]

        # Normalize
        features = F.normalize(features, dim=1)

        # Cosine similarity
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Mask out self-similarity for loss computation
        logits_mask = torch.ones_like(sim_matrix).fill_diagonal_(0)

        # Positive mask: same class & not self
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask * logits_mask  # Remove self-pairs

        # Compute log prob
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log prob of positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


