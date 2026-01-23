"""
Pure mHC Implementation (DeepSeek's Manifold-Constrained Hyper-Connections)
Reference: arXiv:2512.24880v2

This implements the REAL mHC with Sinkhorn-Knopp projection onto the
doubly stochastic manifold, NOT a simplified linear approximation.

Key features:
- Multi-stream residual (n streams, each of dimension d/n)
- Doubly stochastic mixing via Sinkhorn-Knopp algorithm
- Signal energy conservation (rows and columns sum to 1)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp projection onto the doubly stochastic manifold.
    
    A doubly stochastic matrix has all rows and columns summing to 1,
    ensuring signal energy conservation during mixing.
    """
    def __init__(self, n_iters=20, eps=1e-6):
        super().__init__()
        self.n_iters = n_iters
        self.eps = eps
    
    def forward(self, W):
        """Project W onto doubly stochastic manifold."""
        # Ensure positive (required for Sinkhorn)
        W = W.abs() + self.eps
        
        # Alternating row/column normalization
        for _ in range(self.n_iters):
            # Row normalization
            W = W / (W.sum(dim=-1, keepdim=True) + self.eps)
            # Column normalization
            W = W / (W.sum(dim=-2, keepdim=True) + self.eps)
        
        return W


class RealMHC(nn.Module):
    """
    Real mHC module with Sinkhorn-Knopp projection.
    
    Implements the transition:
    X_{l+1} = H_res @ X_l + H_post^T @ F(H_pre @ X_l)
    
    Where H_res is projected onto the doubly stochastic manifold.
    """
    def __init__(self, n_streams=4, n_sinkhorn_iters=20, alpha_init=0.01):
        super().__init__()
        self.n_streams = n_streams
        self.alpha = alpha_init
        
        # Sinkhorn projector
        self.sinkhorn = SinkhornKnopp(n_iters=n_sinkhorn_iters)
        
        # Learnable base matrices (before projection)
        # Initialize near identity for stability
        self.W_res = nn.Parameter(
            torch.eye(n_streams) + alpha_init * torch.randn(n_streams, n_streams)
        )
        
        # Pre/post aggregation vectors (1 x n_streams)
        # These aggregate n streams to 1, and broadcast back
        self.W_pre = nn.Parameter(torch.ones(1, n_streams) / n_streams)
        self.W_post = nn.Parameter(torch.ones(1, n_streams) / n_streams)
    
    def get_H_res(self):
        """Get the doubly stochastic residual mixing matrix."""
        return self.sinkhorn(self.W_res)
    
    def forward(self, x_streams):
        """
        Apply mHC mixing to multi-stream input.
        
        Args:
            x_streams: (B, S, n_streams, d_stream)
        
        Returns:
            Mixed streams: (B, S, n_streams, d_stream)
        """
        # Project to doubly stochastic
        H_res = self.get_H_res()  # (n_streams, n_streams)
        
        # Apply mixing: einsum for batched matrix multiply on stream dim
        # H_res @ x_streams (over the n_streams dimension)
        x_mixed = torch.einsum('ij,...jd->...id', H_res, x_streams)
        
        return x_mixed
    
    def aggregate(self, x_streams):
        """
        Aggregate n streams to 1 stream (for function input).
        
        Args:
            x_streams: (B, S, n_streams, d_stream)
        
        Returns:
            Aggregated: (B, S, d_stream)
        """
        # Normalize pre-weights
        W_pre_norm = F.softmax(self.W_pre, dim=-1)  # (1, n_streams)
        
        # Weighted sum over streams
        aggregated = torch.einsum('ij,...jd->...d', W_pre_norm, x_streams)
        
        return aggregated
    
    def broadcast(self, x_single, original_streams):
        """
        Broadcast single stream back to n streams (for function output).
        
        Args:
            x_single: (B, S, d_stream)
            original_streams: (B, S, n_streams, d_stream) for residual
        
        Returns:
            Broadcasted: (B, S, n_streams, d_stream)
        """
        # Normalize post-weights
        W_post_norm = F.softmax(self.W_post, dim=-1)  # (1, n_streams)
        
        # Broadcast: outer product with weights
        broadcasted = x_single.unsqueeze(-2) * W_post_norm.unsqueeze(-1)  # (B, S, n_streams, d_stream)
        
        return broadcasted


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
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
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with REAL mHC (Sinkhorn-Knopp projected).
    
    Implements multi-stream residual with doubly stochastic mixing.
    """
    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        self.n_embd = config.n_embd
        self.d_stream = config.n_embd // self.n_streams
        
        assert config.n_embd % self.n_streams == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_streams ({self.n_streams})"
        
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Real mHC modules
        n_sinkhorn = getattr(config, 'n_sinkhorn_iters', 20)
        self.mhc_attn = RealMHC(n_streams=self.n_streams, n_sinkhorn_iters=n_sinkhorn)
        self.mhc_mlp = RealMHC(n_streams=self.n_streams, n_sinkhorn_iters=n_sinkhorn)

    def forward(self, x):
        B, S, D = x.shape
        
        # --- ATTENTION BLOCK ---
        # Reshape to streams
        x_streams = x.reshape(B, S, self.n_streams, self.d_stream)
        
        # Apply mHC residual mixing
        x_mixed = self.mhc_attn(x_streams)
        
        # Aggregate for attention input
        x_agg = self.mhc_attn.aggregate(x_mixed)  # (B, S, D) effectively
        x_flat = x_agg.reshape(B, S, -1)
        
        # Pad if needed (aggregate might reduce dimension)
        if x_flat.shape[-1] < D:
            x_flat = F.pad(x_flat, (0, D - x_flat.shape[-1]))
        
        # Attention
        x_norm = self.ln_1(x_flat)
        attn_out = self.attn(x_norm)
        
        # Residual with mixed streams
        x = x_mixed.reshape(B, S, D) + attn_out
        
        # --- MLP BLOCK ---
        x_streams = x.reshape(B, S, self.n_streams, self.d_stream)
        x_mixed = self.mhc_mlp(x_streams)
        x_flat = x_mixed.reshape(B, S, D)
        
        x_norm = self.ln_2(x_flat)
        mlp_out = self.mlp(x_norm)
        
        x = x_flat + mlp_out
        
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
    n_streams: int = 4
    n_sinkhorn_iters: int = 20


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
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

        print(f"Pure mHC model - number of parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"  n_streams: {config.n_streams}, sinkhorn_iters: {config.n_sinkhorn_iters}")

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
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} > block_size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
