"""
E∆-Stream: E∆-MHC-Geo with Per-Stream Compute (Best of Both Worlds)

Combines E∆'s full O(n) geometric operator with JPmHC's per-stream
compute efficiency:
- Geometric operator (Cayley + Householder + gate) on multi-stream state
- Attention and MLP operate at d_stream width (single stream average)
- This allows both wide representation AND deep networks at matched params

Architecture per block:
    1. G_γ(X) — E∆ geometric operator on (B, T, n_embd)
    2. Reshape to (B, T, n, d_stream), average to (B, T, d_stream)
    3. F(LN(avg)) — attention/MLP at d_stream width
    4. Broadcast back and add: X_{l+1} = G_γ(X) + broadcast(F(...))

This matches JPmHC's parameter efficiency while retaining E∆'s unique
full O(n) coverage (Cayley rotation + Householder reflection + learned gate).
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.edelta_hybrid import EdeltaMHCGeoHybrid, LayerNorm


class StreamCausalSelfAttention(nn.Module):
    """Self-attention operating on d_stream dimensions."""

    def __init__(self, d_stream, n_head, dropout=0.0, bias=True, block_size=1024):
        super().__init__()
        self.d_stream = d_stream
        self.n_head = n_head
        assert d_stream % n_head == 0

        self.c_attn = nn.Linear(d_stream, 3 * d_stream, bias=bias)
        self.c_proj = nn.Linear(d_stream, d_stream, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_stream, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class StreamMLP(nn.Module):
    """FFN operating on d_stream dimensions."""

    def __init__(self, d_stream, dropout=0.0, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(d_stream, 4 * d_stream, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_stream, d_stream, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    E∆-Stream Block: Geometric operator + per-stream compute.

    1. Apply E∆ geometric operator G_γ on full n_embd representation
    2. Reshape to streams, average to single d_stream vector
    3. LN → F (attention/MLP at d_stream width) → broadcast back
    4. Residual: X_{l+1} = G_γ(X) + broadcast(F(LN(avg(G_γ(X)))))
    """

    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        self.n_embd = config.n_embd
        self.d_stream = config.n_embd // self.n_streams

        assert config.n_embd % self.n_streams == 0

        # LNs at d_stream width (applied to stream average)
        self.ln_1 = LayerNorm(self.d_stream, bias=config.bias)
        self.ln_2 = LayerNorm(self.d_stream, bias=config.bias)

        # Compute n_head for stream-level attention
        n_head = min(getattr(config, 'n_head', 4), self.d_stream)
        while self.d_stream % n_head != 0:
            n_head -= 1

        # Per-stream attention and MLP
        self.attn = StreamCausalSelfAttention(
            self.d_stream, n_head, config.dropout, config.bias, config.block_size)
        self.mlp = StreamMLP(self.d_stream, config.dropout, config.bias)

        # E∆ geometric operators (full O(n) coverage)
        init_gate_bias = getattr(config, 'init_gate_bias', 0.0)
        gate_reg_weight = getattr(config, 'gate_reg_weight', 0.1)
        geo_hidden_ratio = getattr(config, 'geo_hidden_ratio', 4)

        self.geo_attn = EdeltaMHCGeoHybrid(
            config.n_embd, n_streams=self.n_streams,
            init_gate_bias=init_gate_bias,
            gate_reg_weight=gate_reg_weight,
            geo_hidden_ratio=geo_hidden_ratio)
        self.geo_mlp = EdeltaMHCGeoHybrid(
            config.n_embd, n_streams=self.n_streams,
            init_gate_bias=init_gate_bias,
            gate_reg_weight=gate_reg_weight,
            geo_hidden_ratio=geo_hidden_ratio)

    def forward(self, x):
        B, T, D = x.shape
        n, d = self.n_streams, self.d_stream

        # === ATTENTION SUB-BLOCK ===
        x_geo = self.geo_attn(x)
        x_streams = x_geo.reshape(B, T, n, d)
        x_agg = x_streams.mean(dim=2)      # (B, T, d)
        x_ln = self.ln_1(x_agg)
        y = self.attn(x_ln)                # F at d_stream width
        y_broadcast = y.unsqueeze(2).expand(-1, -1, n, -1).reshape(B, T, D)
        x = x_geo + y_broadcast

        # === MLP SUB-BLOCK ===
        x_geo = self.geo_mlp(x)
        x_streams = x_geo.reshape(B, T, n, d)
        x_agg = x_streams.mean(dim=2)
        x_ln = self.ln_2(x_agg)
        y = self.mlp(x_ln)
        y_broadcast = y.unsqueeze(2).expand(-1, -1, n, -1).reshape(B, T, D)
        x = x_geo + y_broadcast

        return x

    def get_gate_regularization_loss(self):
        return (self.geo_attn.get_gate_regularization_loss() +
                self.geo_mlp.get_gate_regularization_loss())

    def get_gate_statistics(self):
        gamma_attn = getattr(self.geo_attn, '_last_gamma', None)
        gamma_mlp = getattr(self.geo_mlp, '_last_gamma', None)
        return {'gamma_attn': gamma_attn, 'gamma_mlp': gamma_mlp}


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    n_streams: int = 4
    init_gate_bias: float = 0.0
    gate_reg_weight: float = 0.1
    geo_hidden_ratio: int = 4
    use_mhc_projections: bool = True


class GPT(nn.Module):
    """
    E∆-Stream: E∆-MHC-Geo with per-stream compute.

    Combines E∆'s full O(n) geometric operator (Cayley + Householder + gate)
    with JPmHC-style per-stream attention/MLP for parameter efficiency.
    This enables both wide representation and deep networks at matched params.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        d_stream = config.n_embd // config.n_streams
        print(f"E∆-Stream model - parameters: {self.get_num_params() / 1e6:.2f}M")
        print(f"  n_embd: {config.n_embd}, n_streams: {config.n_streams}, d_stream: {d_stream}")
        print(f"  n_layer: {config.n_layer}")
        print(f"  F width: {d_stream} (per-stream, like JPmHC)")
        print(f"  Geometric operator: full O(n) (Cayley + Householder + gate)")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, '_is_gate_net'):
                return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        gate_reg_loss = torch.tensor(0.0, device=device)
        for block in self.transformer.h:
            x = block(x)
            gate_reg_loss = gate_reg_loss + block.get_gate_regularization_loss()

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            loss = loss + gate_reg_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas,
                             device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()
                      if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = (device_type == 'cuda' and
                     'fused' in inspect.signature(
                         torch.optim.AdamW).parameters)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                      betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (idx if idx.size(1) <= self.config.block_size
                        else idx[:, -self.config.block_size:])
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
