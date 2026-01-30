"""
Intelligent Gearbox Hybrid Model (E∆-Gearbox)

This model implements a "switchable transmission" between:
- DDL (Linear Gear): Fast gradients, cheap computation, unstable at large angles
- Cayley (Geodesic Gear): Perfect stability, more expensive, handles any angle

The gate α ∈ (0,1) learns to switch based on input:
    x' = α · DDL(x) + (1-α) · Cayley(x)

THEORETICAL JUSTIFICATION:
==========================

1. DDL (First-Order Approximation):
   Q_ddl ≈ I + 2M
   Error: ||Q_ddl · x||² = ||x||² + O(θ²)
   → Energy grows quadratically with angle

2. Hybrid (Second-Order Approximation):
   Q_hybrid ≈ I - 2M + 2M²  
   Error: ||Q_hybrid · x||² = ||x||² + O(θ⁴)
   → 100x more stable than DDL

3. Cayley (Exact):
   Q = (I+M)^{-1}(I-M)
   Error: ||Q · x||² = ||x||² exactly
   → Perfect energy conservation

THE "INTELLIGENT GEARBOX" BEHAVIOR:
===================================

Small Angles (θ < 15°):
- Linear approximation is accurate
- Gate α → 1 (use DDL)
- Benefit: Faster gradients, cheaper computation

Large Angles (θ > 45°):
- Linear approximation fails catastrophically  
- Gate α → 0 (use Cayley)
- Benefit: Guaranteed stability, no explosion

The model learns this switch automatically by observing:
- DDL causes loss explosion at large angles
- Cayley maintains stable predictions
- Gradient signal teaches gate when to switch

EXPECTED EXPERIMENTAL RESULTS:
==============================

Time to Explosion (||x|| > 2) at θ = 30°:
| Method      | Steps to Explosion | Relative Lifespan |
|-------------|-------------------|-------------------|
| DDL         | ~50               | 1x (baseline)     |
| Hybrid-2nd  | ~5,000            | 100x              |
| Cayley      | ∞                 | Infinite          |
| Gearbox     | ∞ (learns Cayley) | Infinite          |

Author: Arash Shahmansoori (2026)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


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
    # Gearbox-specific
    init_gate_bias: float = 0.0  # 0 = equal mix, positive = favor DDL, negative = favor Cayley


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class GearboxOperator(nn.Module):
    """
    The Intelligent Gearbox: Switches between DDL and Cayley
    
    DDL Mode (α=1):
        x' = x + 2·M·x  where M = β·(uv^T - vu^T)
        Fast but unstable for large rotations
        
    Cayley Mode (α=0):
        x' = Q·x  where Q = (I+M)^{-1}(I-M)
        Slow but perfectly stable for any rotation
        
    Gearbox (0<α<1):
        x' = α·DDL(x) + (1-α)·Cayley(x)
        Best of both worlds - learns when to use each
    """
    
    def __init__(self, n_streams: int, n_embd: int, init_gate_bias: float = 0.0):
        super().__init__()
        self.n_streams = n_streams
        
        # Learnable rotation generator
        self.u = nn.Parameter(torch.randn(n_streams) * 0.02)
        self.v = nn.Parameter(torch.randn(n_streams) * 0.02)
        
        # Rotation magnitude (learnable, entropy-driven)
        self.w_alpha = nn.Parameter(torch.tensor(1.0))
        self.b_init = nn.Parameter(torch.tensor(0.5))
        
        # THE GEARBOX GATE: Learns when to use DDL vs Cayley
        # gate = sigmoid(W_g · mean(x) + b_g)
        # High gate (≈1) = use DDL (fast)
        # Low gate (≈0) = use Cayley (stable)
        self.gate_proj = nn.Linear(n_embd, 1, bias=True)
        
        # Initialize gate bias
        # Positive = favor DDL initially
        # Negative = favor Cayley initially
        # Zero = equal mix
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_gate_bias)
        
        # Cached identity
        self.register_buffer('I', torch.eye(n_streams))
    
    def get_purity_proxy(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Compute entropy proxy Φ = 1 - ||G||_F²/Tr(G)²"""
        G = torch.einsum('bsnd,bsmd->bsnm', x_streams, x_streams)
        frob_sq = torch.einsum('bsij,bsij->bs', G, G)
        trace = torch.einsum('bsii->bs', G)
        trace_sq = trace ** 2
        phi = 1.0 - frob_sq / (trace_sq + 1e-8)
        return phi.mean()
    
    def get_rotation_magnitude(self, x_streams: torch.Tensor) -> torch.Tensor:
        """Thermodynamic gating: β = softplus(w_α · Φ + b)"""
        phi = self.get_purity_proxy(x_streams)
        beta = F.softplus(self.w_alpha * phi + self.b_init)
        return beta
    
    def forward(self, x: torch.Tensor, x_streams: torch.Tensor) -> tuple:
        """
        Apply the Gearbox transformation.
        
        Args:
            x: Full representation (B, S, D) for gate computation
            x_streams: Stream representation (B, S, n, d) for transformation
            
        Returns:
            x_transformed: (B, S, n, d)
            gate_value: Scalar for logging/analysis
        """
        B, S, n, d = x_streams.shape
        
        # === COMPUTE GATE (input-dependent) ===
        # Use sequence-pooled representation for gate decision
        x_pooled = x.mean(dim=1)  # (B, D)
        gate_logit = self.gate_proj(x_pooled)  # (B, 1)
        gate = torch.sigmoid(gate_logit)  # (B, 1)
        
        # === COMPUTE ROTATION MAGNITUDE ===
        beta = self.get_rotation_magnitude(x_streams)
        
        # === CONSTRUCT SKEW-SYMMETRIC GENERATOR ===
        A = torch.outer(self.u, self.v) - torch.outer(self.v, self.u)
        M = (beta / 2) * A
        
        # === DDL OUTPUT (fast, linear) ===
        # x_ddl = x + 2Mx = (I + 2M)x
        x_ddl = x_streams + 2 * torch.einsum('ij,bsjd->bsid', M, x_streams)
        
        # === CAYLEY OUTPUT (stable, geodesic) ===
        # Q = (I + M)^{-1}(I - M)
        I_plus_M = self.I + M
        I_minus_M = self.I - M
        Q = torch.linalg.solve(I_plus_M, I_minus_M)
        x_cayley = torch.einsum('ij,bsjd->bsid', Q, x_streams)
        
        # === GEARBOX MIXING ===
        # Expand gate for broadcasting: (B, 1) -> (B, 1, 1, 1)
        gate_expanded = gate.view(B, 1, 1, 1)
        x_gearbox = gate_expanded * x_ddl + (1 - gate_expanded) * x_cayley
        
        return x_gearbox, gate.mean().item()


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
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
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


class GearboxBlock(nn.Module):
    """Transformer block with Intelligent Gearbox operator."""

    def __init__(self, config):
        super().__init__()
        self.n_streams = config.n_streams
        self.d_stream = config.n_embd // config.n_streams
        
        # Gearbox operators (one for attention, one for MLP)
        self.gearbox_attn = GearboxOperator(
            config.n_streams, config.n_embd, config.init_gate_bias
        )
        self.gearbox_mlp = GearboxOperator(
            config.n_streams, config.n_embd, config.init_gate_bias
        )
        
        # Standard transformer components
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # For logging gate values
        self.last_gate_attn = 0.5
        self.last_gate_mlp = 0.5

    def forward(self, x):
        B, S, D = x.shape
        
        # === ATTENTION BLOCK ===
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        x_transformed, gate_attn = self.gearbox_attn(x, x_streams)
        self.last_gate_attn = gate_attn
        x = x_transformed.reshape(B, S, D)
        x = x + self.attn(self.ln_1(x))
        
        # === MLP BLOCK ===
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        x_transformed, gate_mlp = self.gearbox_mlp(x, x_streams)
        self.last_gate_mlp = gate_mlp
        x = x_transformed.reshape(B, S, D)
        x = x + self.mlp(self.ln_2(x))
        
        return x


class GPT(nn.Module):
    """GPT with Intelligent Gearbox (E∆-Gearbox)"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([GearboxBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"E∆-Gearbox model: {self.get_num_params() / 1e6:.2f}M parameters")

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

    def get_gate_values(self):
        """Return gate values for all layers (for analysis)."""
        gates = []
        for i, block in enumerate(self.transformer.h):
            gates.append({
                'layer': i,
                'gate_attn': block.last_gate_attn,
                'gate_mlp': block.last_gate_mlp
            })
        return gates

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
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
