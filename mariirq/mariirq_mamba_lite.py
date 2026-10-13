import torch
import torch.nn as nn

from .bidirectional_mamba2 import BidirectionalMambaBlock
from .utils import get_1d_sincos_pos_embed


class LightPatchEmbed(nn.Module):
    def __init__(self, num_rows=4, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(num_rows, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.proj(x)))


class MariIRQMambaLite(nn.Module):
    def __init__(
        self,
        num_classes,
        num_rows=4,
        max_matrix_len=250,
        embed_dim=192,
        depth=3,
        headdim=32,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_patches = max_matrix_len

        self.patch_embed = LightPatchEmbed(num_rows=num_rows, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_matrix_len + 1, embed_dim), requires_grad=False
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(
                    layer_idx=i,
                    d_model=embed_dim,
                    headdim=headdim,
                    d_state=64,
                    d_conv=4,
                    expand=2,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
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

    def forward(self, x, idx=None):
        x = self.patch_embed(x).squeeze(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x)

        return self.fc(self.fc_norm(x[:, 0, :]))


CountMambaLight = MariIRQMambaLite
