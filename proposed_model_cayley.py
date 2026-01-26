"""
Pure Cayley Rotation Model (E∆-Cayley)

This model uses ONLY the Cayley transform for geometric transformations.
No Householder reflection - just pure rotation on SO(n).

MATHEMATICAL PROPERTIES:
========================

1. PERFECT ISOMETRY:
   ||Qx|| = ||x||  for all x
   
   Proof: Q = (I+M)^{-1}(I-M) where M is skew-symmetric
          Q^T Q = (I-M)^T(I+M)^{-T}(I+M)^{-1}(I-M)
                = (I+M)(I-M)^{-1}(I+M)^{-1}(I-M)
                = I  (by commutativity of polynomials in M)

2. UNCONDITIONAL STABILITY:
   det(I + M) ≠ 0  for all skew-symmetric M
   
   Proof: Eigenvalues of (I + M) are 1 + iμ where μ ∈ ℝ
          |1 + iμ| = √(1 + μ²) ≥ 1 > 0

3. EIGENVALUE RANGE:
   λ(Q) = e^{-2i·arctan(βμ/2)}  where iμ = eigenvalue of A
   
   Since arctan: ℝ → (-π/2, π/2), we have:
   arg(λ) ∈ (-π, π), STRICTLY EXCLUDING ±π
   
   CONSEQUENCE: λ = -1 is IMPOSSIBLE
   This is why pure Cayley cannot negate - it preserves orientation.

4. DETERMINANT:
   det(Q) = +1  always (proper rotation, not reflection)

USE CASE:
=========
- Geometric reasoning tasks (rotation, coordinate transforms)
- Long-term stability requirements (deep networks, long sequences)
- Energy-conserving dynamics simulation

NOT SUITABLE FOR:
=================
- Correction tasks ("Actually, no") - cannot negate information
- Rapid belief revision - only gradual rotation

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
    n_streams: int = 4  # Number of parallel streams for rotation


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CayleyRotation(nn.Module):
    """
    Pure Cayley Rotation Operator
    
    Maps skew-symmetric generator A to rotation matrix Q ∈ SO(n):
        Q = (I + βA/2)^{-1} (I - βA/2)
    
    Properties:
    - Q^T Q = I (orthogonal)
    - det(Q) = +1 (proper rotation)
    - ||Qx|| = ||x|| (isometry)
    
    The rotation magnitude β can be:
    - Fixed (init_beta): For stable, consistent rotation
    - Entropy-driven: Larger rotation when uncertain (thermodynamic gating)
    """
    
    def __init__(self, n_streams: int, use_entropy_gating: bool = True):
        super().__init__()
        self.n_streams = n_streams
        self.use_entropy_gating = use_entropy_gating
        
        # Learnable rotation generator: A = uv^T - vu^T (skew-symmetric)
        self.u = nn.Parameter(torch.randn(n_streams) * 0.02)
        self.v = nn.Parameter(torch.randn(n_streams) * 0.02)
        
        # Identity matrix (cached for efficiency)
        self.register_buffer('I', torch.eye(n_streams))
        
        if use_entropy_gating:
            # Thermodynamic gating: β = softplus(w_α · Φ + b)
            self.w_alpha = nn.Parameter(torch.tensor(1.0))
            self.b_init = nn.Parameter(torch.tensor(0.0))
        else:
            # Fixed rotation magnitude
            self.beta = nn.Parameter(torch.tensor(1.0))
    
    def get_purity_proxy(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        Compute Frobenius Purity Proxy (Φ)
        
        Φ = 1 - ||G||_F² / Tr(G)²
        
        where G = X X^T is the Gram matrix.
        
        Returns:
            Φ ∈ [0, 1-1/n]: Higher = more mixed (uncertain)
        """
        # x_streams: (B, S, n, d)
        # Compute Gram matrix G = X X^T: (B, S, n, n)
        G = torch.einsum('bsnd,bsmd->bsnm', x_streams, x_streams)
        
        # Frobenius norm squared
        frob_sq = torch.einsum('bsij,bsij->bs', G, G)
        
        # Trace squared
        trace = torch.einsum('bsii->bs', G)
        trace_sq = trace ** 2
        
        # Purity proxy
        phi = 1.0 - frob_sq / (trace_sq + 1e-8)
        
        return phi.mean()  # Average over batch and sequence
    
    def get_rotation_matrix(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute Cayley rotation matrix Q = (I + M)^{-1}(I - M)
        where M = β/2 · A and A = uv^T - vu^T
        
        Args:
            beta: Rotation magnitude scalar
            
        Returns:
            Q: Rotation matrix in SO(n)
        """
        # Construct skew-symmetric generator
        A = torch.outer(self.u, self.v) - torch.outer(self.v, self.u)
        
        # Scale by beta/2
        M = (beta / 2) * A
        
        # Cayley transform: Q = (I + M)^{-1}(I - M)
        I_plus_M = self.I + M
        I_minus_M = self.I - M
        
        # Solve (I + M) Q = (I - M)
        Q = torch.linalg.solve(I_plus_M, I_minus_M)
        
        return Q, A
    
    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        Apply Cayley rotation to stream representation.
        
        Args:
            x_streams: (B, S, n_streams, d_stream)
            
        Returns:
            x_rotated: (B, S, n_streams, d_stream)
        """
        # Compute rotation magnitude
        if self.use_entropy_gating:
            phi = self.get_purity_proxy(x_streams)
            beta = F.softplus(self.w_alpha * phi + self.b_init)
        else:
            beta = F.softplus(self.beta)
        
        # Get rotation matrix
        Q, _ = self.get_rotation_matrix(beta)
        
        # Apply rotation: x' = Q @ x
        x_rotated = torch.einsum('ij,bsjd->bsid', Q, x_streams)
        
        return x_rotated


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention."""

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
    """Standard MLP block."""

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


class CayleyBlock(nn.Module):
    """
    Transformer block with Pure Cayley rotation.
    
    Structure:
        x_streams = reshape(x) to (B, S, n, d)
        x_rotated = Cayley(x_streams)
        x = reshape(x_rotated) to (B, S, D)
        x = x + Attention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(self, config):
        super().__init__()
        self.n_streams = config.n_streams
        self.d_stream = config.n_embd // config.n_streams
        
        # Cayley rotation
        self.cayley_attn = CayleyRotation(config.n_streams, use_entropy_gating=True)
        self.cayley_mlp = CayleyRotation(config.n_streams, use_entropy_gating=True)
        
        # Standard transformer components
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        B, S, D = x.shape
        
        # === ATTENTION BLOCK ===
        # Reshape to streams
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        
        # Apply Cayley rotation
        x_rotated = self.cayley_attn(x_streams)
        
        # Reshape back
        x = x_rotated.reshape(B, S, D)
        
        # Standard attention with residual
        x = x + self.attn(self.ln_1(x))
        
        # === MLP BLOCK ===
        # Reshape to streams
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        
        # Apply Cayley rotation
        x_rotated = self.cayley_mlp(x_streams)
        
        # Reshape back
        x = x_rotated.reshape(B, S, D)
        
        # Standard MLP with residual
        x = x + self.mlp(self.ln_2(x))
        
        return x


class GPT(nn.Module):
    """GPT with Pure Cayley Rotation (E∆-Cayley)"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([CayleyBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"E∆-Cayley model: {self.get_num_params() / 1e6:.2f}M parameters")

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
