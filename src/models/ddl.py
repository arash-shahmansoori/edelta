"""
Pure DDL Implementation (Deep Delta Learning)
Reference: arXiv:2601.00417v1

This implements the Deep Delta Learning operator with Householder-like
rank-1 perturbation of the identity.

Key features:
- Delta operator: A(X) = I - β(X) * kk^T / (k^T k + ε)
- Learnable reflection direction k(X)
- Learnable gating scalar β(X) ∈ [0, 2]
- WARNING: Singular at β=1 (this is a known limitation!)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class DeepDeltaOperator(nn.Module):
    """
    Deep Delta Learning operator (Zhang et al., 2026).
    
    Applies a rank-1 perturbation to the identity:
    A(X) = I - β * kk^T / (k^T k + ε)
    
    Where:
    - k(X) is the learned reflection direction
    - β(X) is the learned gate ∈ [0, 2]
    
    WARNING: This operator is SINGULAR when β=1!
    The matrix (I - kk^T) has a zero eigenvalue.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Network to compute reflection direction k
        self.k_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        
        # Network to compute gate β
        self.beta_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Scale factor for β (maps [0,1] to [0,2])
        self.beta_scale = 2.0
        
    def forward(self, x, return_beta=False):
        """
        Apply Deep Delta operator.
        
        Args:
            x: Input tensor (B, S, D)
            return_beta: If True, also return beta values
        
        Returns:
            Transformed x (B, S, D)
            Optionally: beta values (B, S, 1)
        """
        B, S, D = x.shape
        
        # Compute pooled representation for k and β
        x_pooled = x.mean(dim=1)  # (B, D)
        
        # Compute reflection direction k (normalized)
        k = self.k_net(x_pooled)  # (B, D)
        k = F.normalize(k, dim=-1)  # Unit norm
        
        # Compute gate β ∈ [0, 2]
        beta = self.beta_scale * self.beta_net(x_pooled)  # (B, 1)
        
        # Delta operator: A = I - β * kk^T / (k^T k + ε)
        # Since k is normalized, k^T k = 1
        # So A = I - β * kk^T
        
        # Apply to x: Ax = x - β * k * (k^T @ x)
        # k: (B, D), x: (B, S, D)
        k_expanded = k.unsqueeze(1)  # (B, 1, D)
        
        # Dot product: k^T @ x for each position
        kTx = (k_expanded * x).sum(dim=-1, keepdim=True)  # (B, S, 1)
        
        # Delta update: x - β * k * (k^T x)
        beta_expanded = beta.unsqueeze(1)  # (B, 1, 1)
        delta = beta_expanded * k_expanded * kTx  # (B, S, D)
        
        x_out = x - delta
        
        if return_beta:
            return x_out, beta.unsqueeze(1).expand(B, S, 1)
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
    Transformer block with Deep Delta Learning operator.
    
    The Delta operator applies a rank-1 geometric transformation
    to the residual stream, allowing reflections/projections.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Deep Delta operators
        self.delta_attn = DeepDeltaOperator(config.n_embd)
        self.delta_mlp = DeepDeltaOperator(config.n_embd)

    def forward(self, x):
        # --- ATTENTION BLOCK ---
        # Apply Delta operator to residual
        x_delta, beta_attn = self.delta_attn(x, return_beta=True)
        
        # Standard attention on transformed input
        attn_out = self.attn(self.ln_1(x_delta))
        
        # Residual connection (DDL style: add to delta-transformed)
        x = x_delta + attn_out
        
        # --- MLP BLOCK ---
        x_delta, beta_mlp = self.delta_mlp(x, return_beta=True)
        
        mlp_out = self.mlp(self.ln_2(x_delta))
        x = x_delta + mlp_out
        
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

        print(f"Pure DDL model - number of parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"  WARNING: DDL has singularity at β=1!")

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
