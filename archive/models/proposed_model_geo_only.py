"""
Geodesic-only model: Rotation WITHOUT mHC mixing matrices.
Used to isolate the effect of rotation vs mixing.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeodesicDelta(nn.Module):
    """Geodesic rotation operator (same as proposed_model.py)"""
    def __init__(self, d_model, n_streams=4, init_bias=-6.0):
        super().__init__()
        assert d_model % n_streams == 0
        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams

        self.u = nn.Parameter(torch.randn(1, 1, n_streams, 1) * 0.01)
        self.v = nn.Parameter(torch.randn(1, 1, n_streams, 1) * 0.01)
        self.w_alpha = nn.Parameter(torch.zeros(1))
        self.b_init = nn.Parameter(torch.tensor(init_bias))
        self.register_buffer('I', torch.eye(n_streams).view(1, 1, n_streams, n_streams))

    def get_purity_proxy(self, x_streams):
        G = torch.matmul(x_streams, x_streams.transpose(-1, -2))
        frob_sq = (G * G).sum(dim=(-1, -2), keepdim=True)
        tr = torch.einsum('...ii->...', G).unsqueeze(-1).unsqueeze(-1)
        tr_sq = tr * tr + 1e-6
        return 1.0 - (frob_sq / tr_sq)

    def forward(self, x, return_beta=False):
        B, S, D = x.shape
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        
        phi = self.get_purity_proxy(x_streams)
        beta = F.softplus(self.w_alpha * phi + self.b_init)
        
        A = torch.matmul(self.u, self.v.transpose(-1, -2)) - torch.matmul(self.v, self.u.transpose(-1, -2))
        M = beta * A
        
        # Taylor approximation for speed
        M2 = torch.matmul(M, M)
        Q = self.I - 2*M + 2*M2
        
        x_rotated = torch.matmul(Q, x_streams)
        x_out = x_rotated.view(B, S, D)
        
        if return_beta:
            return x_out, beta.squeeze(-1)
        return x_out


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                              dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Block with Geodesic rotation but NO mHC mixing matrices."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Geodesic rotation ONLY (no mHC mixing)
        self.geo_attn = GeodesicDelta(config.n_embd, n_streams=4, init_bias=-6.0)
        self.geo_mlp = GeodesicDelta(config.n_embd, n_streams=4, init_bias=-6.0)
        self.use_damper = getattr(config, 'use_damper', True)

    def forward(self, x):
        # Attention block with rotation
        x_rotated, beta = self.geo_attn(x, return_beta=True)
        damper = 1.0 - torch.tanh(beta) if self.use_damper else 1.0
        x = x_rotated + damper * self.attn(self.ln_1(x_rotated))  # NO mixing matrices
        
        # MLP block with rotation
        x_rotated, beta = self.geo_mlp(x, return_beta=True)
        damper = 1.0 - torch.tanh(beta) if self.use_damper else 1.0
        x = x_rotated + damper * self.mlp(self.ln_2(x_rotated))  # NO mixing matrices
        
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_damper: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, geo_lr_mult=1.0):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        geo_params = [p for pn, p in param_dict.items() if 'geo_' in pn]
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2 and 'geo_' not in pn]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2 and 'geo_' not in pn]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
        ]
        if geo_params:
            optim_groups.append({'params': geo_params, 'weight_decay': weight_decay, 'lr': learning_rate * geo_lr_mult})
            print(f"num geodesic params: {len(geo_params)}, LR x{geo_lr_mult}")
        
        print(f"num decayed: {len(decay_params)}, num non-decayed: {len(nodecay_params)}")
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        return torch.optim.AdamW(optim_groups, betas=betas, fused=fused)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError()

    def crop_block_size(self, block_size):
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
