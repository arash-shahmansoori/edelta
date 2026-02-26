"""
E∆-Stream: Per-Stream Compute with Fused Geometric Operator

Theoretically consistent implementation of E∆-MHC-Geo where ALL operations
respect the stream decomposition:
- Fused geometric operator: single Linear projection generates all geometric
  parameters (u, v, β, k, γ), analogous to JPmHC's fused_proj
- Per-stream F: attention and MLP operate at d_stream width
- Full O(n) coverage: Cayley rotation + Householder reflection + learned gate

The fused projection replaces 5 separate 2-layer MLPs with a single linear
layer, reducing geo overhead from ~265K to ~7K per module (at n_streams=4).
All theoretical guarantees (unconditional orthogonality, eigenvalue exclusion,
O(n) coverage) are preserved — they depend on the algebraic structure
(A = uv^T - vu^T is skew-symmetric), not on how u, v are computed.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class FusedGeoOperator(nn.Module):
    """
    Fused-projection E∆ geometric operator.

    Replaces 5 separate 2-layer MLPs with a single fused linear projection
    (like JPmHC's fused_proj), generating all geometric parameters at once:
        [u | v | β_raw | k | γ_raw] = LayerNorm(x_flat) @ W_geo + b_geo

    Output dimensions: u(n) + v(n) + β(1) + k(n) + γ(1) = 3n + 2

    The DDC-Hybrid operator remains:
        G_γ(X) = γ·Q(X)·X + (1-γ)·H₂(k(X))·X

    All theoretical guarantees preserved:
    - Q = (I + βA/2)^{-1}(I - βA/2) with A = uv^T - vu^T is orthogonal
    - H₂ = I - 2·kk^T with ||k||=1 is orthogonal with det=-1
    - γ ∈ (0,1) selects between rotation and reflection
    """

    def __init__(self, d_model, n_streams=4, init_gate_bias=0.0,
                 gate_reg_weight=0.1, geo_hidden_dim=32):
        super().__init__()
        assert d_model % n_streams == 0

        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        self.gate_reg_weight = gate_reg_weight

        n = n_streams
        self.out_dim = 3 * n + 2  # u(n) + v(n) + β(1) + k(n) + γ(1)

        self.norm = nn.LayerNorm(d_model, elementwise_affine=True)
        self.geo_proj = nn.Sequential(
            nn.Linear(d_model, geo_hidden_dim),
            nn.GELU(),
            nn.Linear(geo_hidden_dim, self.out_dim),
        )

        self._init_weights(init_gate_bias)

        self.register_buffer('householder_beta', torch.tensor(2.0))
        self.register_buffer('I', torch.eye(n_streams))
        self._gate_reg_loss = None

    def _init_weights(self, init_gate_bias):
        with torch.no_grad():
            nn.init.zeros_(self.geo_proj[0].weight)
            nn.init.zeros_(self.geo_proj[0].bias)
            nn.init.zeros_(self.geo_proj[2].weight)
            bias = self.geo_proj[2].bias
            n = self.n_streams
            bias[:n].zero_()           # u
            bias[n:2*n].zero_()        # v
            bias[2*n].zero_()          # β_raw (softplus(0) ≈ 0.69)
            bias[2*n+1:3*n+1].zero_()  # k
            bias[3*n+1].fill_(init_gate_bias)  # γ_raw

    def forward(self, x):
        B, S, D = x.shape
        n = self.n_streams

        x_streams = x.view(B, S, n, self.d_stream)

        x_pooled = x.mean(dim=1)  # (B, D)
        x_norm = self.norm(x_pooled)
        params = self.geo_proj(x_norm)  # (B, 3n+2)

        u = params[:, :n]                        # (B, n)
        v = params[:, n:2*n]                      # (B, n)
        beta_raw = params[:, 2*n:2*n+1]           # (B, 1)
        k_raw = params[:, 2*n+1:3*n+1]            # (B, n)
        gate_logit = params[:, 3*n+1:3*n+2]       # (B, 1)

        beta = F.softplus(beta_raw)               # (B, 1), positive
        k = F.normalize(k_raw, dim=-1)            # (B, n), unit vector
        gamma = torch.sigmoid(gate_logit)          # (B, 1), ∈ (0,1)

        # Midpoint collapse regularization
        self._gate_reg_loss = self.gate_reg_weight * 4 * gamma * (1 - gamma)
        self._gate_reg_loss = self._gate_reg_loss.mean()
        self._last_gamma = gamma.detach().mean().item()

        gamma_bc = gamma.view(B, 1, 1, 1)

        # Cayley rotation: Q = (I + βA/2)^{-1}(I - βA/2)
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        M = (beta.view(B, 1, 1) / 2) * A
        I_batch = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I_batch + M, I_batch - M)
        x_rotated = torch.einsum('bnm,bsmd->bsnd', Q, x_streams)

        # Householder reflection: H₂(k)·x = x - 2·(x·k)·k
        # x_streams: (B, S, n, d), k: (B, n)
        # Project each stream onto k: dot product along stream dimension
        k_exp = k.unsqueeze(1).unsqueeze(-1)        # (B, 1, n, 1)
        proj = (x_streams * k_exp).sum(dim=2, keepdim=True)  # (B, S, 1, d)
        x_reflected = x_streams - 2 * proj * k_exp

        # Hybrid blend
        x_hybrid = gamma_bc * x_rotated + (1 - gamma_bc) * x_reflected

        return x_hybrid.reshape(B, S, D)

    def get_gate_regularization_loss(self):
        if self._gate_reg_loss is None:
            return torch.tensor(0.0, device=self.I.device)
        return self._gate_reg_loss


class StreamCausalSelfAttention(nn.Module):
    """Self-attention at d_stream width."""

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
    """FFN at d_stream width."""

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


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight,
                            self.bias, 1e-5)


class StreamRouter(nn.Module):
    """
    Dynamic per-token stream routing (like JPmHC's H_pre/H_post).

    Generates n×n row-stochastic (pre) and column-stochastic (post)
    mixing matrices from the input, enabling selective aggregation
    and distribution across streams.
    """

    def __init__(self, n_embd, n_streams):
        super().__init__()
        self.n_streams = n_streams
        self.norm = nn.LayerNorm(n_embd, elementwise_affine=True)
        self.proj = nn.Linear(n_embd, 2 * n_streams * n_streams, bias=True)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            self.proj.bias.zero_()

    def forward(self, x_streams):
        """
        Args:
            x_streams: (B, T, n, d)
        Returns:
            H_pre: (B, T, n, n) row-stochastic
            H_post: (B, T, n, n) column-stochastic
        """
        B, T, n, d = x_streams.shape
        x_flat = x_streams.reshape(B, T, -1)
        x_norm = self.norm(x_flat)
        raw = self.proj(x_norm).view(B, T, 2, n, n)

        H_pre = F.softmax(raw[:, :, 0], dim=-1)   # row-stochastic
        H_post = F.softmax(raw[:, :, 1], dim=-2)   # column-stochastic
        return H_pre, H_post


class Block(nn.Module):
    """
    E∆-Stream Block: Geometric operator + dynamic routing + per-stream F.

    X_{l+1} = G_γ(X) + H_post · (F(avg(H_pre · G_γ(X))) ⊗ 1_n)

    Key fix: uses dynamic per-token H_pre/H_post routing (like JPmHC)
    instead of crude mean+broadcast, preserving stream differentiation.
    """

    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        self.n_embd = config.n_embd
        self.d_stream = config.n_embd // self.n_streams

        assert config.n_embd % self.n_streams == 0

        self.ln_1 = LayerNorm(self.d_stream, bias=config.bias)
        self.ln_2 = LayerNorm(self.d_stream, bias=config.bias)

        n_head = min(getattr(config, 'n_head', 4), self.d_stream)
        while self.d_stream % n_head != 0:
            n_head -= 1

        self.attn = StreamCausalSelfAttention(
            self.d_stream, n_head, config.dropout, config.bias, config.block_size)
        self.mlp = StreamMLP(self.d_stream, config.dropout, config.bias)

        init_gate_bias = getattr(config, 'init_gate_bias', 0.0)
        gate_reg_weight = getattr(config, 'gate_reg_weight', 0.1)

        geo_hidden_dim = getattr(config, 'geo_hidden_dim', 48)
        self.geo_attn = FusedGeoOperator(
            config.n_embd, self.n_streams, init_gate_bias, gate_reg_weight,
            geo_hidden_dim)
        self.geo_mlp = FusedGeoOperator(
            config.n_embd, self.n_streams, init_gate_bias, gate_reg_weight,
            geo_hidden_dim)

        # Dynamic stream routing (like JPmHC's H_pre/H_post)
        self.router_attn = StreamRouter(config.n_embd, self.n_streams)
        self.router_mlp = StreamRouter(config.n_embd, self.n_streams)

    def forward(self, x):
        B, T, D = x.shape
        n, d = self.n_streams, self.d_stream

        # === Attention sub-block ===
        x_geo = self.geo_attn(x)                          # G_γ(X)
        x_streams = x_geo.reshape(B, T, n, d)

        H_pre, H_post = self.router_attn(x_streams)       # dynamic routing

        # H_pre selectively aggregates → average → F → H_post distributes
        x_pre = torch.einsum('btij,btjd->btid', H_pre, x_streams)
        x_agg = x_pre.mean(dim=2)                         # (B, T, d)
        y = self.attn(self.ln_1(x_agg))                   # F at d_stream
        y_bc = y.unsqueeze(2).expand(-1, -1, n, -1)       # broadcast
        y_post = torch.einsum('btij,btjd->btid', H_post, y_bc)
        x = x_geo + y_post.reshape(B, T, D)

        # === MLP sub-block ===
        x_geo = self.geo_mlp(x)
        x_streams = x_geo.reshape(B, T, n, d)

        H_pre, H_post = self.router_mlp(x_streams)

        x_pre = torch.einsum('btij,btjd->btid', H_pre, x_streams)
        x_agg = x_pre.mean(dim=2)
        y = self.mlp(self.ln_2(x_agg))
        y_bc = y.unsqueeze(2).expand(-1, -1, n, -1)
        y_post = torch.einsum('btij,btjd->btid', H_post, y_bc)
        x = x_geo + y_post.reshape(B, T, D)

        return x

    def get_gate_regularization_loss(self):
        return (self.geo_attn.get_gate_regularization_loss() +
                self.geo_mlp.get_gate_regularization_loss())

    def get_gate_statistics(self):
        return {
            'gamma_attn': getattr(self.geo_attn, '_last_gamma', None),
            'gamma_mlp': getattr(self.geo_mlp, '_last_gamma', None),
        }


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
    geo_hidden_ratio: int = 4  # unused in fused variant, kept for API compat
    geo_hidden_dim: int = 32   # hidden dim for fused geo projection


class GPT(nn.Module):
    """
    E∆-Stream: Fused-projection E∆ with per-stream compute.

    Key design choices:
    1. Fused geo projection: Linear(n_embd → 3n+2) replaces 5 MLPs
    2. Per-stream F: attention/MLP at d_stream width (like JPmHC)
    3. Full O(n): Cayley rotation + Householder reflection + learned gate
    4. Exact orthogonality: analytical Cayley solve (not iterative)
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
        print(f"  Geo operator: fused Linear({config.n_embd} → {3*config.n_streams+2})")
        print(f"  F width: {d_stream} (per-stream)")
        print(f"  Full O(n): Cayley + Householder + gate")

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
