"""Portable Mamba-1 with per-lead temporal processing and explicit lead mixing.

Equations/initialization follow state-spaces/mamba's mamba_simple.py and
selective_scan_ref. This implementation uses only differentiable PyTorch ops.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def affine_scan(a, b):
    """Inclusive affine recurrence with zero initial state, [B,T,D,N]."""
    state = b.new_zeros(b.shape[0], b.shape[2], b.shape[3])
    outputs = []
    for start in range(0, b.shape[1], 64):
        aa, bb = a[:, start:start + 64], b[:, start:start + 64]
        offset = 1
        while offset < bb.shape[1]:
            bb = torch.cat((bb[:, :offset], bb[:, offset:] + aa[:, offset:] * bb[:, :-offset]), 1)
            aa = torch.cat((aa[:, :offset], aa[:, offset:] * aa[:, :-offset]), 1)
            offset *= 2
        states = bb + aa * state.unsqueeze(1)
        outputs.append(states)
        state = states[:, -1]
    return torch.cat(outputs, 1)


class _SelectiveScan(torch.autograd.Function):
    """Analytic first-order backward avoids retaining prefix-scan intermediates.

    This reduces memory without changing the recurrence. Higher-order gradients
    are intentionally unsupported; ordinary model training uses first derivatives.
    """
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):
        a = torch.exp(delta.unsqueeze(-1) * A)
        b = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        h = affine_scan(a, b)
        ctx.save_for_backward(u, delta, A, B, C, D, h)
        return (h * C.unsqueeze(2)).sum(-1) + u * D

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        u, delta, A, B, C, D, h = ctx.saved_tensors
        a = torch.exp(delta.unsqueeze(-1) * A)
        # Adjoint: lambda[t] = grad[t]*C[t] + a[t+1]*lambda[t+1].
        shifted = torch.cat([a[:, 1:], torch.zeros_like(a[:, :1])], 1)
        adjoint = affine_scan(shifted.flip(1), (grad.unsqueeze(-1) * C.unsqueeze(2)).flip(1)).flip(1)
        previous = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
        ga = adjoint * previous * a
        grad_u = (adjoint * delta.unsqueeze(-1) * B.unsqueeze(2)).sum(-1) + grad * D
        grad_delta = (ga * A + adjoint * B.unsqueeze(2) * u.unsqueeze(-1)).sum(-1)
        grad_A = (ga * delta.unsqueeze(-1)).sum((0, 1))
        grad_B = (adjoint * delta.unsqueeze(-1) * u.unsqueeze(-1)).sum(2)
        grad_C = (grad.unsqueeze(-1) * h).sum(2)
        grad_D = (grad * u).sum((0, 1))
        return grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D


def selective_scan(u, delta, A, B, C, D):
    """Mamba-1 selective recurrence, using portable PyTorch operations only.

    u/delta [batch,time,inner], B/C [batch,time,state], A [inner,state].
    Chunked associative scans avoid unstable division by cumulative products.
    """
    return _SelectiveScan.apply(u, delta, A, B, C, D)


class Mamba1(nn.Module):
    def __init__(self, d_model=32, d_state=16, expand=2, d_conv=4):
        super().__init__()
        self.inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner, self.inner, d_conv,
                              groups=self.inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner)
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))
        dt, b, c = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        y = selective_scan(u, delta, -self.A_log.exp(), b, c, self.D)
        return self.out_proj(y * F.silu(z))


class BidirectionalMamba(nn.Module):
    def __init__(self, d_model=32, dropout=0.2, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_branch = Mamba1(d_model=d_model, **kwargs)
        self.backward_branch = Mamba1(d_model=d_model, **kwargs)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        forward = self.forward_branch(h)
        backward = self.backward_branch(h.flip(1)).flip(1)
        return x + self.dropout(self.merge(torch.cat([forward, backward], dim=-1)))


class ECGMamba(nn.Module):
    def __init__(self, d_model=32, d_state=16, expand=2, d_conv=4,
                 n_blocks=2, dropout=0.2, mix_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 7, stride=2, padding=3)
        self.norm1 = nn.LayerNorm(16)
        self.conv2 = nn.Conv1d(16, d_model, 5, stride=2, padding=2)
        self.norm2 = nn.LayerNorm(d_model)
        self.blocks = nn.Sequential(*[BidirectionalMamba(
            d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv,
            dropout=dropout) for _ in range(n_blocks)])
        self.mixer = nn.Sequential(nn.Linear(12 * d_model, mix_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(2 * mix_dim, 1)

    def encode_leads(self, x):
        if x.ndim != 3 or x.shape[1] != 12 or x.shape[2] < 1:
            raise ValueError('Expected ECG [batch,12,time] with nonempty time dimension')
        batch = x.shape[0]
        h = x.reshape(batch * 12, 1, x.shape[-1])
        h = self.norm1(F.gelu(self.conv1(h)).transpose(1, 2))
        h = self.norm2(F.gelu(self.conv2(h.transpose(1, 2))).transpose(1, 2))
        h = self.blocks(h)
        return h.reshape(batch, 12, h.shape[1], h.shape[2])

    def forward(self, x):
        h = self.encode_leads(x)
        h = h.permute(0, 2, 1, 3).flatten(2)
        h = self.mixer(h)
        return self.classifier(torch.cat([h.mean(1), h.amax(1)], dim=-1))
