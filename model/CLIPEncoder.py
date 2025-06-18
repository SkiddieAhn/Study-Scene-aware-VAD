import torch
import torch.nn as nn
import clip

class CLIPEncoder(nn.Module):
    def __init__(self, network='ViT-L/14'):
        super().__init__()
        self.clip_model, _ = clip.load(network, device='cuda')
        self.visual_model = self.clip_model.visual.float()

        for p in self.visual_model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.visual_model(x)  # [B*T, D]
        return feat.view(B, T, -1)