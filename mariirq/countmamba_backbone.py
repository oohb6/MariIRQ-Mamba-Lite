# Adapted from the CountMamba-WF backbone. See README.md for attribution.

import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from timm.layers import DropPath

from .utils import get_1d_sincos_pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(1, 1))

    def forward(self, x):
        return self.proj(x)


class CausalCNN(nn.Module):
    def __init__(self, in_channels, mid_channel, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, mid_channel, kernel_size, padding=kernel_size - 1)
        self.bn1 = nn.BatchNorm1d(mid_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels, mid_channel, kernel_size, padding=kernel_size - 1)
        self.bn2 = nn.BatchNorm1d(mid_channel)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(in_channels, mid_channel, kernel_size, padding=kernel_size - 1)
        self.bn3 = nn.BatchNorm1d(mid_channel)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels, mid_channel, kernel_size, padding=kernel_size - 1)
        self.bn4 = nn.BatchNorm1d(mid_channel)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x.squeeze(2)

        x = self.relu1(self.bn1(self.conv1(x)[:, :, : -(self.kernel_size - 1)]))
        x = self.relu2(self.bn2(self.conv2(x)[:, :, : -(self.kernel_size - 1)]))
        x = self.dropout1(self.pool1(x))

        x = self.relu3(self.bn3(self.conv3(x)[:, :, : -(self.kernel_size - 1)]))
        x = self.relu4(self.bn4(self.conv4(x)[:, :, : -(self.kernel_size - 1)]))
        x = self.dropout2(self.pool2(x))

        return x.transpose(1, 2)


class MariIRQAdapt(nn.Module):
    def __init__(
        self,
        num_classes,
        drop_path_rate=0.2,
        embed_dim=256,
        depth=3,
        patch_size=35,
        max_matrix_len=4998,
    ):
        super().__init__()
        if max_matrix_len % 6 != 0:
            raise ValueError("max_matrix_len must be divisible by 6 for the CausalCNN downsampler")

        num_patches = max_matrix_len // 6
        self.num_patches = num_patches

        self.patch_embed = PatchEmbed((patch_size, 1), in_chans=1, embed_dim=embed_dim)
        self.local_model = CausalCNN(embed_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [Mamba2(layer_idx=i, d_model=embed_dim, headdim=embed_dim // 4) for i in range(depth)]
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList(
            [DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity() for i in range(depth)]
        )

        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        weight = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, idx):
        x = self.patch_embed(x)
        x = self.local_model(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block, drop_path in zip(self.blocks, self.droppaths):
            x = drop_path(block(x)) + x

        x = self.fc_norm(x)
        x = x[:, 1:, :]

        aggregate_idx = torch.floor(idx / 6).long().clamp(max=x.shape[1] - 1)
        pooled = torch.stack([x[i, : aggregate_idx[i] + 1].mean(dim=0) for i in range(x.size(0))])
        return self.fc(pooled)


CountMambaModel = MariIRQAdapt
