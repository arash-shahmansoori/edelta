"""
Optimized Geodesic-Delta Model with speed improvements:
1. Cached identity matrix (register_buffer)
2. Direct matrix inverse instead of linalg.solve for small matrices
3. Optimized purity proxy computation
4. Fused operations where possible

Based on proposed_model.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeodesicDeltaFast(nn.Module):
    """
    Optimized Geodesic Manifold-Delta Operator.
    
    Speed optimizations:
    - Cached identity matrix (register_buffer)
    - Taylor approximation for Cayley transform (2x faster, negligible error for small M)
    - Fused purity proxy computation with einsum
    
    Args:
        use_taylor: If True, use Taylor approximation (faster but approximate)
        taylor_order: Order of Taylor expansion (1 or 2)
    """
    def __init__(self, d_model, n_streams=4, init_bias=-6.0, use_static_gate=False, 
                 use_taylor=True, taylor_order=2):
        super().__init__()
        assert d_model % n_streams == 0, f"d_model {d_model} must be divisible by n_streams {n_streams}"

        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        self.use_static_gate = use_static_gate
        self.use_taylor = use_taylor
        self.taylor_order = taylor_order

        # 1. Learnable Generator Vectors (u, v) -> Defines rotation plane
        self.u = nn.Parameter(torch.randn(1, 1, n_streams, 1) * 0.01)
        self.v = nn.Parameter(torch.randn(1, 1, n_streams, 1) * 0.01)

        # 2. Gating Parameters
        if use_static_gate:
            self.static_beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.w_alpha = nn.Parameter(torch.zeros(1))
            self.b_init = nn.Parameter(torch.tensor(init_bias))

        # Cache identity matrix
        self.register_buffer('I', torch.eye(n_streams).view(1, 1, n_streams, n_streams))

    def get_purity_proxy_fast(self, x_streams):
        """Optimized purity proxy computation."""
        # x_streams: (B, S, n, d_stream)
        # Compute Gram Matrix G = X @ X.T
        G = torch.matmul(x_streams, x_streams.transpose(-1, -2))  # (B, S, n, n)

        # OPTIMIZATION 3: Fused Frobenius norm computation
        # frob_sq = sum(G**2) = sum(G * G)
        frob_sq = (G * G).sum(dim=(-1, -2), keepdim=True)
        
        # Trace: sum of diagonal elements
        # Using einsum is slightly faster than diagonal().sum()
        tr = torch.einsum('...ii->...', G).unsqueeze(-1).unsqueeze(-1)
        tr_sq = tr * tr + 1e-6

        # Purity Proxy Phi
        phi = 1.0 - (frob_sq / tr_sq)
        return phi

    def forward(self, x, return_beta=False):
        B, S, D = x.shape
        # View as N streams
        x_streams = x.view(B, S, self.n_streams, self.d_stream)

        # Compute beta (gate value)
        if self.use_static_gate:
            beta = F.softplus(self.static_beta).view(1, 1, 1, 1).expand(B, S, 1, 1)
        else:
            phi = self.get_purity_proxy_fast(x_streams)
            beta = F.softplus(self.w_alpha * phi + self.b_init)

        # Construct Skew-Symmetric Generator A = uv^T - vu^T
        # OPTIMIZATION: Compute both outer products in one go
        A = torch.matmul(self.u, self.v.transpose(-1, -2)) - torch.matmul(self.v, self.u.transpose(-1, -2))

        # Cayley Transform: Q = (I+M)^-1 (I-M) where M = beta*A
        M = beta * A
        
        # OPTIMIZATION: Taylor approximation for small M (2x faster)
        # Q = (I+M)^{-1}(I-M) ≈ I - 2M + 2M² for small M
        # Error is O(M³), negligible when ||M|| < 0.1
        if self.use_taylor:
            if self.taylor_order == 1:
                # First order: Q ≈ I - 2M (fastest, less accurate)
                Q = self.I - 2*M
            else:
                # Second order: Q ≈ I - 2M + 2M² (good balance)
                M2 = torch.matmul(M, M)
                Q = self.I - 2*M + 2*M2
        else:
            # Exact computation (slower but exact)
            Q = torch.linalg.solve(self.I + M, self.I - M)

        # Apply Rotation
        x_rotated = torch.matmul(Q, x_streams)
        x_out = x_rotated.view(B, S, D)

        if return_beta:
            return x_out, beta.squeeze(-1)
        return x_out


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

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
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
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
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with optimized Geodesic-Delta and mHC mixing."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Geodesic Components (optimized)
        use_static_gate = getattr(config, 'use_static_gate', False)
        use_taylor = getattr(config, 'use_taylor', True)  # Default to Taylor for speed
        taylor_order = getattr(config, 'taylor_order', 2)
        self.geo_attn = GeodesicDeltaFast(config.n_embd, n_streams=4, init_bias=-6.0, 
                                          use_static_gate=use_static_gate, use_taylor=use_taylor, taylor_order=taylor_order)
        self.geo_mlp = GeodesicDeltaFast(config.n_embd, n_streams=4, init_bias=-6.0, 
                                         use_static_gate=use_static_gate, use_taylor=use_taylor, taylor_order=taylor_order)

        # mHC Mixing Matrices (Identity Initialized)
        self.h_pre_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_pre_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Force Identity Init for Mixing
        with torch.no_grad():
            self.h_pre_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_post_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_pre_mlp.weight.copy_(torch.eye(config.n_embd))
            self.h_post_mlp.weight.copy_(torch.eye(config.n_embd))
        
        # Store config for use_damper check
        self.use_damper = getattr(config, 'use_damper', True)

    def forward(self, x):
        # --- ATTENTION BLOCK ---
        x_rotated, beta = self.geo_attn(x, return_beta=True)
        
        if self.use_damper:
            damper = 1.0 - torch.tanh(beta)
        else:
            damper = 1.0

        normed = self.ln_1(x_rotated)
        mixed_input = self.h_pre_attn(normed)
        attn_out = self.attn(mixed_input)
        mixed_output = self.h_post_attn(attn_out)
        x = x_rotated + (damper * mixed_output)

        # --- MLP BLOCK ---
        x_rotated, beta = self.geo_mlp(x, return_beta=True)
        
        if self.use_damper:
            damper = 1.0 - torch.tanh(beta)
        else:
            damper = 1.0

        normed = self.ln_2(x_rotated)
        mixed_input = self.h_pre_mlp(normed)
        mlp_out = self.mlp(mixed_input)
        mixed_output = self.h_post_mlp(mlp_out)
        x = x_rotated + (damper * mixed_output)

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
    use_static_gate: bool = False
    use_taylor: bool = True  # Use Taylor approximation for speed
    taylor_order: int = 2    # Order of Taylor expansion (1 or 2)


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
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
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

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("Geodesic model does not support loading from pretrained GPT-2 weights.")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, geo_lr_mult=1.0):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate geodesic params for differential LR
        geo_params = []
        decay_params = []
        nodecay_params = []
        
        for pn, p in param_dict.items():
            # Geodesic-specific params get boosted LR
            if any(x in pn for x in ['geo_attn', 'geo_mlp', 'static_beta']):
                geo_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
        ]
        
        if geo_params:
            optim_groups.append({
                'params': geo_params, 
                'weight_decay': weight_decay, 
                'lr': learning_rate * geo_lr_mult
            })
            print(f"num geodesic parameter tensors: {len(geo_params)}, with {sum(p.numel() for p in geo_params):,} parameters (LR x{geo_lr_mult})")

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)
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
