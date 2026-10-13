import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


def _align_mamba_dims(d_inner, d_state, headdim, align_to=8):
    def projected_size(value):
        return 2 * value + 2 * d_state + value // headdim

    if projected_size(d_inner) % align_to == 0:
        return d_inner, d_inner // headdim

    for step in range(headdim, 2048, headdim):
        for candidate in (d_inner + step, d_inner - step):
            if candidate <= 0 or candidate % headdim != 0:
                continue
            if projected_size(candidate) % align_to == 0:
                return candidate, candidate // headdim

    raise ValueError(f"Unable to align d_inner={d_inner} for headdim={headdim}")


class BidirectionalMambaBlock(nn.Module):
    def __init__(
        self,
        layer_idx,
        d_model=192,
        headdim=32,
        d_state=64,
        d_conv=4,
        expand=2,
        drop_path=0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.forward_mamba = _Mamba2Core(d_model, d_state, d_conv, expand, headdim)
        self.reverse_mamba = _Mamba2Core(d_model, d_state, d_conv, expand, headdim)
        self.fusion_proj = nn.Linear(2 * d_model, d_model)

        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[-2].bias, 2.0)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        normalized = self.norm1(x)

        forward_out = self.forward_mamba(normalized)
        reversed_input = torch.flip(normalized, dims=[1])
        reverse_out = torch.flip(self.reverse_mamba(reversed_input), dims=[1])

        fused = F.gelu(self.fusion_proj(torch.cat([forward_out, reverse_out], dim=-1)))
        fused = self.norm2(fused)
        gate = self.gate_net(normalized)
        return residual + self.drop_path(gate * fused)


class _Mamba2Core(nn.Module):
    def __init__(self, d_model=192, d_state=64, d_conv=4, expand=2, headdim=32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim

        raw_d_inner = expand * d_model
        self.d_inner, self.nheads = _align_mamba_dims(raw_d_inner, d_state, headdim)
        self._build_mamba_ssm()

    def _build_mamba_ssm(self):
        try:
            from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
        except ImportError as exc:
            raise ImportError(
                "MariIRQ-Mamba-Lite requires mamba-ssm. See README.md for the tested environment."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        factory_kwargs = {"device": device, "dtype": torch.float32}

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=self.d_conv - 1,
            bias=True,
            **factory_kwargs,
        )

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=0.0001)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(1, 1.1)
        self.A_log = nn.Parameter(torch.log(A).float())
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.norm = RMSNormGated(
            self.d_inner,
            eps=1e-5,
            norm_before_gate=False,
            group_size=self.d_inner,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, u):
        from einops import rearrange
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log.float())
        return mamba_split_conv1d_scan_combined(
            zxbcdt,
            rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias,
            self.dt_bias,
            A,
            D=self.D,
            chunk_size=256,
            seq_idx=None,
            activation="silu",
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.eps,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.headdim,
            ngroups=1,
            norm_before_gate=False,
        )
