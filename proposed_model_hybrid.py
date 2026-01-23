"""
E∆-Hybrid: True Hybrid Geodesic Operator (Cayley Rotation + Householder Reflection)

This model combines:
1. Cayley ROTATION (SO(n)) - for geometric tasks, preserves all information
2. Householder REFLECTION - for corrections/negation, can flip information
3. Learnable GATE - model learns when to rotate vs reflect

Mathematical Insight:
- Cayley rotation: All eigenvalues on unit circle → cannot negate
- Householder reflection: Eigenvalue -1 along k → can negate
- Hybrid: Best of both worlds!

Expected Benefits:
- Geometric tasks (rotation2d, rotation3d): Use rotation (isometric)
- Correction tasks ("Aha!" moments): Use reflection (negation)
- Unified architecture handles ALL task types
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeodesicDeltaHybrid(nn.Module):
    """
    True Hybrid Operator: Cayley Rotation + Householder Reflection
    
    The model learns WHEN to rotate vs reflect via a learnable gate.
    
    Gate interpretation:
    - gate → 1: Use rotation (geometric tasks)
    - gate → 0: Use reflection (correction tasks)
    """
    def __init__(self, d_model, n_streams=4, init_gate_bias=0.0):
        super().__init__()
        assert d_model % n_streams == 0
        
        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        
        # === CAYLEY ROTATION PARAMS ===
        # Rotation plane defined by (u, v)
        self.u = nn.Parameter(torch.randn(n_streams, 1) * 0.1)
        self.v = nn.Parameter(torch.randn(n_streams, 1) * 0.1)
        
        # Rotation magnitude control
        self.rot_w_alpha = nn.Parameter(torch.ones(1))
        self.rot_b_init = nn.Parameter(torch.tensor(0.0))
        
        # === HOUSEHOLDER REFLECTION PARAMS ===
        # Reflection direction (will be normalized)
        self.k_raw = nn.Parameter(torch.randn(n_streams) * 0.1)
        
        # Reflection magnitude (β in DDL paper, typically 0-2)
        self.ref_scale = nn.Parameter(torch.tensor(1.0))
        
        # === HYBRID GATE ===
        # Learns when to rotate vs reflect based on input
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_gate_bias)
        
        # Cache identity matrix
        self.register_buffer('I', torch.eye(n_streams))
    
    def get_purity_proxy(self, x_streams):
        """Frobenius Purity Proxy for rotation magnitude"""
        G = torch.matmul(x_streams, x_streams.transpose(-1, -2))
        frob_sq = torch.sum(G**2, dim=(-1, -2), keepdim=True)
        tr = torch.einsum('...ii->...', G).unsqueeze(-1).unsqueeze(-1)
        tr_sq = tr ** 2 + 1e-6
        phi = 1.0 - (frob_sq / tr_sq)
        return phi
    
    def cayley_rotation(self, x_streams, beta):
        """Apply Cayley rotation with magnitude β"""
        # Skew-symmetric generator
        A = self.u @ self.v.T - self.v @ self.u.T  # (n_streams, n_streams)
        
        # Scale by β
        A_scaled = beta.mean() * A
        
        # Cayley transform: Q = (I - A/2)^{-1}(I + A/2)
        M = A_scaled / 2
        Q = torch.linalg.solve(self.I + M, self.I - M)
        
        # Apply rotation
        x_rotated = torch.einsum('ij,bsjd->bsid', Q, x_streams)
        return x_rotated
    
    def householder_reflection(self, x_streams):
        """Apply Householder reflection along learned direction k"""
        # Normalize k to unit vector
        k = F.normalize(self.k_raw, dim=0)  # (n_streams,)
        
        # Reflection magnitude (clamped to [0, 2] like DDL)
        beta_ref = torch.sigmoid(self.ref_scale) * 2.0
        
        # Householder: H = I - β(k ⊗ k)
        # Applied: x' = x - β(x·k)k
        # For streams: project along stream dimension
        
        # x_streams: (B, S, n_streams, d_stream)
        # k: (n_streams,) → (1, 1, n_streams, 1)
        k_expanded = k.view(1, 1, self.n_streams, 1)
        
        # Dot product: (x · k) for each position in d_stream
        dot = (x_streams * k_expanded).sum(dim=2, keepdim=True)  # (B, S, 1, d_stream)
        
        # Reflection: x - β * (x·k) * k
        x_reflected = x_streams - beta_ref * dot * k_expanded
        
        return x_reflected
    
    def forward(self, x, return_debug=False):
        B, S, D = x.shape
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        
        # === COMPUTE GATE (rotation vs reflection) ===
        # Use input-dependent gating
        x_pooled = x.mean(dim=1)  # (B, D)
        gate = torch.sigmoid(self.gate_proj(x_pooled))  # (B, 1)
        gate = gate.view(B, 1, 1, 1)  # Broadcast shape
        
        # === CAYLEY ROTATION ===
        phi = self.get_purity_proxy(x_streams)
        beta_rot = F.softplus(self.rot_w_alpha * phi + self.rot_b_init)
        x_rotated = self.cayley_rotation(x_streams, beta_rot)
        
        # === HOUSEHOLDER REFLECTION ===
        x_reflected = self.householder_reflection(x_streams)
        
        # === HYBRID OUTPUT ===
        # gate=1 → rotation, gate=0 → reflection
        x_hybrid = gate * x_rotated + (1 - gate) * x_reflected
        
        x_out = x_hybrid.reshape(B, S, D)
        
        if return_debug:
            return x_out, {
                'gate': gate.squeeze(),
                'beta_rot': beta_rot.mean(),
                'phi': phi.mean(),
            }
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
    Hybrid Block: Applies GeodesicDeltaHybrid (rotation OR reflection)
    followed by standard attention/MLP.
    
    Pattern: x → Hybrid(x) → +Attention → Hybrid(x) → +MLP
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Hybrid operators (rotation + reflection)
        init_gate_bias = getattr(config, 'init_gate_bias', 0.0)
        self.hybrid_attn = GeodesicDeltaHybrid(
            config.n_embd, n_streams=4, init_gate_bias=init_gate_bias
        )
        self.hybrid_mlp = GeodesicDeltaHybrid(
            config.n_embd, n_streams=4, init_gate_bias=init_gate_bias
        )

    def forward(self, x):
        # Attention block: Hybrid transform → Attention → Residual
        x_hybrid = self.hybrid_attn(x)
        x = x_hybrid + self.attn(self.ln_1(x_hybrid))
        
        # MLP block: Hybrid transform → MLP → Residual
        x_hybrid = self.hybrid_mlp(x)
        x = x_hybrid + self.mlp(self.ln_2(x_hybrid))
        
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
    # Hybrid specific
    init_gate_bias: float = 0.0  # 0 = equal rotation/reflection, >0 = prefer rotation


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

        print(f"E∆-Hybrid model - number of parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"  init_gate_bias: {config.init_gate_bias}")
        print(f"  (gate=1 → rotation, gate=0 → reflection)")

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
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

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
