"""
Geodesic-Delta Model V2: Fixed rotation generator collapse.

Key change: u, v vectors are normalized to unit norm, preventing collapse.
The rotation magnitude is controlled ONLY by β, not by ||u||, ||v||.
"""

import math
import inspect
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Global flag to enable/disable beta diagnostics
ENABLE_BETA_LOGGING = False  # Disabled to avoid torch.compile issues
BETA_LOG_PROB = 0.01


class GeodesicDeltaV2(nn.Module):
    """
    Geodesic Manifold-Delta Operator V2.
    
    Key fix: u, v are NORMALIZED to unit vectors before computing A.
    This prevents the model from disabling rotation by zeroing u, v.
    Rotation magnitude is controlled ONLY by β.
    """
    def __init__(self, d_model, n_streams=4, init_bias=-6.0, use_static_gate=False):
        super().__init__()
        assert d_model % n_streams == 0, f"d_model {d_model} must be divisible by n_streams {n_streams}"

        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        self.use_static_gate = use_static_gate

        # Learnable rotation plane vectors - will be NORMALIZED before use
        # Initialize orthogonally for maximum rotation capacity
        self.u_raw = nn.Parameter(torch.randn(n_streams) * 0.1)
        self.v_raw = nn.Parameter(torch.randn(n_streams) * 0.1)

        # Gating Parameters
        if use_static_gate:
            self.static_beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.w_alpha = nn.Parameter(torch.zeros(1))
            self.b_init = nn.Parameter(torch.tensor(init_bias))

    def get_normalized_generators(self):
        """Get unit-normalized u, v vectors with numerical stability."""
        u_norm = torch.norm(self.u_raw).clamp(min=1e-6)
        v_norm = torch.norm(self.v_raw).clamp(min=1e-6)
        u = self.u_raw / u_norm
        v = self.v_raw / v_norm
        return u, v

    def get_purity_proxy(self, x_streams):
        # x_streams: (B, S, n, d_stream)
        G = torch.matmul(x_streams, x_streams.transpose(-1, -2))
        frob_sq = torch.sum(G**2, dim=(-1, -2), keepdim=True)
        tr = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        tr_sq = tr ** 2 + 1e-6
        phi = 1.0 - (frob_sq / tr_sq)
        return phi

    def forward(self, x, return_beta=False):
        B, S, D = x.shape
        x_streams = x.view(B, S, self.n_streams, self.d_stream)

        # Compute beta with clamping for numerical stability
        if self.use_static_gate:
            beta = F.softplus(self.static_beta).clamp(max=10.0)
            beta = beta.view(1, 1, 1, 1).expand(B, S, 1, 1)
        else:
            phi = self.get_purity_proxy(x_streams)
            beta = F.softplus(self.w_alpha * phi + self.b_init).clamp(max=10.0)

        # Get NORMALIZED generators (KEY FIX!)
        u, v = self.get_normalized_generators()
        
        # Reshape for broadcasting
        u = u.view(1, 1, self.n_streams, 1)
        v = v.view(1, 1, self.n_streams, 1)

        # Construct skew-symmetric generator A = uv^T - vu^T
        A = torch.matmul(u, v.transpose(-1, -2)) - torch.matmul(v, u.transpose(-1, -2))

        # Cayley Transform: Q = (I+βA)^{-1}(I-βA)
        M = beta * A
        I = torch.eye(self.n_streams, device=x.device).view(1, 1, self.n_streams, self.n_streams)
        Q = torch.linalg.solve(I + M, I - M)

        # Apply Rotation
        x_rotated = torch.matmul(Q, x_streams)
        x_out = x_rotated.view(B, S, D)

        # Diagnostic logging
        if ENABLE_BETA_LOGGING and self.training and random.random() < BETA_LOG_PROB:
            u_norm = torch.norm(self.u_raw).item()
            v_norm = torch.norm(self.v_raw).item()
            A_norm = torch.norm(A).item()
            rotation_mag = torch.norm(x_out - x).mean().item()
            print(f"[BETA-V2] β: {beta.mean().item():.6f} | ||u_raw||: {u_norm:.4f} | "
                  f"||v_raw||: {v_norm:.4f} | ||A||: {A_norm:.4f} | ||Qx-x||: {rotation_mag:.6f}")

        if return_beta:
            return x_out, beta.squeeze(-1)
        return x_out


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


class BlockV2(nn.Module):
    """Transformer block with GeodesicDeltaV2 (fixed rotation collapse)."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        self.use_damper = getattr(config, 'use_damper', True)
        use_static_gate = getattr(config, 'use_static_gate', False)

        # V2 Geodesic components with normalized generators
        self.geo_attn = GeodesicDeltaV2(config.n_embd, n_streams=4, init_bias=-6.0, 
                                         use_static_gate=use_static_gate)
        self.geo_mlp = GeodesicDeltaV2(config.n_embd, n_streams=4, init_bias=-6.0,
                                        use_static_gate=use_static_gate)

        # mHC Mixing Matrices (Identity Initialized)
        self.h_pre_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_pre_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)

        with torch.no_grad():
            self.h_pre_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_post_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_pre_mlp.weight.copy_(torch.eye(config.n_embd))
            self.h_post_mlp.weight.copy_(torch.eye(config.n_embd))

    def forward(self, x):
        # Attention block
        x_rotated, beta = self.geo_attn(x, return_beta=True)
        normed = self.ln_1(x_rotated)
        mixed_input = self.h_pre_attn(normed)
        attn_out = self.attn(mixed_input)
        mixed_output = self.h_post_attn(attn_out)

        if self.use_damper:
            damper = 1.0 - torch.tanh(beta)
            x = x_rotated + (damper * mixed_output)
        else:
            x = x_rotated + mixed_output

        # MLP block
        x_rotated, beta = self.geo_mlp(x, return_beta=True)
        normed = self.ln_2(x_rotated)
        mixed_input = self.h_pre_mlp(normed)
        mlp_out = self.mlp(mixed_input)
        mixed_output = self.h_post_mlp(mlp_out)

        if self.use_damper:
            damper = 1.0 - torch.tanh(beta)
            x = x_rotated + (damper * mixed_output)
        else:
            x = x_rotated + mixed_output

        return x


@dataclass
class GPTConfigV2:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    use_damper: bool = True
    use_static_gate: bool = False
    geo_lr_mult: float = 50.0


class GPTV2(nn.Module):
    """GPT with GeodesicDeltaV2 (fixed rotation collapse)."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([BlockV2(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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
        assert t <= self.config.block_size
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
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, geo_lr_mult=50.0):
        geo_params = []
        decay_params = []
        nodecay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in ['geo_', 'u_raw', 'v_raw', 'w_alpha', 'b_init', 'static_beta']):
                geo_params.append(param)
            elif param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': geo_params, 'weight_decay': 0.0, 'lr': learning_rate * geo_lr_mult}
        ]
        
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        num_geo = sum(p.numel() for p in geo_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")
        print(f"num geodesic parameter tensors: {len(geo_params)}, with {num_geo:,} parameters (LR x{geo_lr_mult})")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

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
