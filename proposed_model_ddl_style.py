"""
E∆-DDL-Style: Geodesic Manifold-Delta with DDL-style Application Pattern

Key changes from original E∆-MHC-Geo:
1. β controls ROTATION MAGNITUDE (not damper)
2. DDL-style application: Transform FIRST, then standard residual
3. No bypass possible - rotation always applies
4. Removed damper term entirely

This combines:
- Cayley rotation (better than Householder: isometric, preserves information)
- DDL's structural pattern (proven 136x better on corrections)
- Thermodynamic gating (β from purity proxy)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeodesicDeltaDDLStyle(nn.Module):
    """
    Cayley Rotation with DDL-style application.
    
    Key difference from original:
    - β directly scales the rotation magnitude
    - No damper term - rotation always applies
    - Larger β = larger rotation (can approach 180° for corrections)
    """
    def __init__(self, d_model, n_streams=4, init_bias=0.0, rotation_scale=1.0):
        super().__init__()
        assert d_model % n_streams == 0, f"d_model {d_model} must be divisible by n_streams {n_streams}"

        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        self.rotation_scale = rotation_scale

        # Learnable Generator Vectors (u, v) -> Defines rotation plane
        # Initialize with small values but not too small
        self.u = nn.Parameter(torch.randn(n_streams, 1) * 0.1)
        self.v = nn.Parameter(torch.randn(n_streams, 1) * 0.1)

        # Thermodynamic Gating Parameters
        # init_bias=0 means β starts at softplus(0) ≈ 0.69 (moderate rotation)
        self.w_alpha = nn.Parameter(torch.ones(1))  # Start with weight=1
        self.b_init = nn.Parameter(torch.tensor(init_bias))
        
        # Cache identity matrix
        self.register_buffer('I', torch.eye(n_streams))

    def get_purity_proxy(self, x_streams):
        """Compute Frobenius Purity Proxy Φ"""
        # x_streams: (B, S, n_streams, d_stream)
        # Gram Matrix G = X @ X.T
        G = torch.matmul(x_streams, x_streams.transpose(-1, -2))  # (B, S, n, n)

        # Frobenius Norm squared
        frob_sq = torch.sum(G**2, dim=(-1, -2), keepdim=True)  # (B, S, 1, 1)
        
        # Trace using einsum (faster than diagonal().sum())
        tr = torch.einsum('...ii->...', G).unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
        tr_sq = tr ** 2 + 1e-6

        # Purity Proxy: Φ = 1 - ||G||_F² / (Tr(G))²
        phi = 1.0 - (frob_sq / tr_sq)
        return phi  # (B, S, 1, 1)

    def forward(self, x, return_beta=False):
        B, S, D = x.shape
        
        # View as N streams
        x_streams = x.view(B, S, self.n_streams, self.d_stream)

        # === COMPUTE β (rotation magnitude) ===
        phi = self.get_purity_proxy(x_streams)  # (B, S, 1, 1)
        beta = F.softplus(self.w_alpha * phi + self.b_init)  # (B, S, 1, 1)
        
        # β now directly controls rotation magnitude
        # Scale β to allow larger rotations
        beta_scaled = beta * self.rotation_scale

        # === CONSTRUCT ROTATION MATRIX ===
        # Skew-Symmetric Generator: A = uv^T - vu^T
        A = self.u @ self.v.T - self.v @ self.u.T  # (n_streams, n_streams)
        
        # Scale A by β (KEY CHANGE: β controls rotation magnitude)
        # Use mean β across batch/sequence for the rotation matrix
        # This ensures Q is the same for all positions (simpler, more stable)
        beta_mean = beta_scaled.mean()
        A_scaled = beta_mean * A

        # === CAYLEY TRANSFORM ===
        # Q = (I - A/2)^{-1} (I + A/2)  [using scaled A]
        # Equivalent to: Q = (I + M)^{-1} (I - M) where M = A_scaled/2
        M = A_scaled / 2
        I = self.I
        
        # Solve for numerical stability
        Q = torch.linalg.solve(I + M, I - M)

        # === APPLY ROTATION (ALWAYS - no bypass) ===
        x_rotated = torch.einsum('ij,bsjd->bsid', Q, x_streams)
        x_out = x_rotated.reshape(B, S, D)

        if return_beta:
            return x_out, beta.squeeze(-1).squeeze(-1)  # (B, S)
        return x_out


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
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
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
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
    DDL-style Transformer Block with Cayley Rotation.
    
    Pattern: Transform FIRST, then standard residual
    x → Rotate → LayerNorm → Attention → + → Rotate → LayerNorm → MLP → +
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Geodesic rotation operators (DDL-style)
        rotation_scale = getattr(config, 'rotation_scale', 1.0)
        self.geo_attn = GeodesicDeltaDDLStyle(
            config.n_embd, n_streams=4, init_bias=0.0, rotation_scale=rotation_scale
        )
        self.geo_mlp = GeodesicDeltaDDLStyle(
            config.n_embd, n_streams=4, init_bias=0.0, rotation_scale=rotation_scale
        )

    def forward(self, x):
        # === ATTENTION BLOCK (DDL-style) ===
        # 1. Rotate FIRST (always applies, no bypass)
        x_rotated = self.geo_attn(x)
        
        # 2. Standard residual on rotated input
        x = x_rotated + self.attn(self.ln_1(x_rotated))
        
        # === MLP BLOCK (DDL-style) ===
        # 1. Rotate FIRST
        x_rotated = self.geo_mlp(x)
        
        # 2. Standard residual on rotated input
        x = x_rotated + self.mlp(self.ln_2(x_rotated))
        
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
    # DDL-style specific
    rotation_scale: float = 1.0  # Scale factor for β (larger = stronger rotations)


class GPT(nn.Module):
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
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"E∆-DDL-Style model - number of parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"  rotation_scale: {config.rotation_scale}")

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
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block_size {self.config.block_size}"
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

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate decay and no-decay params
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

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
