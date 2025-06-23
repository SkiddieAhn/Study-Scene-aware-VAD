import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # [B, T, D]


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, batch_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z  # [B, T, input_dim]


class SubconAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, num_layers=2, nhead=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, input_dim))  # Max T=512

        self.encoder = TransformerEncoder(input_dim, hidden_dim, nhead, num_layers)
        self.decoder = TransformerDecoder(input_dim, hidden_dim, nhead, num_layers)

        self.proj_to_latent = nn.Linear(input_dim, latent_dim)
        self.recon_to_org = nn.Linear(latent_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pos_embedding[:, :T, :]  # [B, T, D]

        # encoding
        enc_out = self.encoder(x)  # [B, T, D]
        z = self.proj_to_latent(enc_out)  # [B, T, latent_dim]

        # decoding
        z_prime = self.recon_to_org(z)  # [B, T, D]
        dec = self.decoder(z_prime)  # [B, T, D]
        recon = self.output_proj(dec)  # [B, T, D]

        # classification
        sfeature = z_prime.mean(dim=1)  # [B, D]
        logits = self.classifier(sfeature)  # [B, num_classes]

        return recon, logits, sfeature
