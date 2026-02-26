"""
JPmHC Implementation (JP Morgan Hyper-Connections with Cayley Retraction)
Reference: Sengupta, Wang & Brunswic, arXiv:2602.18308, February 2026

Faithful to the paper's architecture:

Architecture (Parallel Routing, their Equation 14):
    x_out = H_res · x_streams + H_post · (y ⊗ 1_n)
    where y = F(avg(H_pre · x_streams))

F operates on a SINGLE d-dimensional stream-averaged vector (Section 3.2:
"F is the sub-layer, evaluated once on a single p-dimensional vector").

Coefficient Computation (their Equation 12/35):
    [H̃_pre | H̃_post | H̃_res] = W_fused · LayerNorm(x_flat)

Constraints:
    H_pre  = softmax(H̃_pre, dim=-1)        → row-stochastic
    H_post = softmax(H̃_post, dim=-2)       → column-stochastic
    H_res  = IterativeCayley(H̃_res - H̃_res^T)  → approx orthogonal

Iterative Cayley (their Algorithm 8, Appendix I.2):
    W = H̃_res - H̃_res^T  (skew-symmetrize)
    Y_0 = I + αW            (initialization, their Eq. 31)
    Y_{i+1} = I + (α/2) W (I + Y_i),  i = 0, ..., s-1  (their Eq. 32)
    with α = 0.1, s = 2 iterations

Key differences from E∆-MHC-Geo:
    - Parallel routing (vs series pre-transformation)
    - Full-rank Cayley generator (vs rank-2)
    - Iterative approximation (vs exact analytical solve)
    - SO(n) only (vs full O(n) with Householder gate)
    - Per-token dynamic pre/post (vs learned weight matrices)
    - Sub-layer F on single stream d (vs full n_embd)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def iterative_cayley(H_tilde, alpha=0.1, n_iters=2):
    """
    Iterative fixed-point Cayley retraction (JPmHC Algorithm 8).

    Given unconstrained H̃, skew-symmetrizes to W = H̃ − H̃^T,
    then approximates the Cayley retraction via fixed-point iteration.

    Initialization (their Eq. 31):  Y₀ = I + αW
    Iteration (their Eq. 32):       Y_{i+1} = I + (α/2) W(I + Y_i)

    With s=2, α=0.1 this achieves ||Y^T Y - I||_max < 10^{-3}
    (their Proposition I.1).

    Args:
        H_tilde: (*, n, n) unconstrained matrix
        alpha: Step size (default 0.1 per their paper)
        n_iters: Number of iterations s (default 2 per their paper)

    Returns:
        Y: (*, n, n) approximately orthogonal matrix
    """
    W = H_tilde - H_tilde.transpose(-1, -2)

    n = W.shape[-1]
    I = torch.eye(n, device=W.device, dtype=W.dtype)

    Y = I + alpha * W
    for _ in range(n_iters):
        Y = I + (alpha / 2) * torch.matmul(W, I + Y)

    return Y


class JPmHCModule(nn.Module):
    """
    JPmHC dynamic routing module with per-token matrix generation.

    Generates all three n×n mixing matrices (H_pre, H_post, H_res)
    dynamically per token via a single fused linear projection (Eq. 12/35),
    then applies parallel routing (Eq. 14):
        x_out = H_res·x + H_post·(F(avg(H_pre·x)) ⊗ 1_n)

    F is evaluated on a single d-dimensional stream-averaged vector,
    per the paper's Section 3.2 and Figure 2.
    """

    def __init__(self, n_streams=4, n_embd=None, cayley_alpha=0.1,
                 cayley_iters=2):
        super().__init__()
        self.n_streams = n_streams
        self.n_embd = n_embd
        self.d_stream = n_embd // n_streams
        self.cayley_alpha = cayley_alpha
        self.cayley_iters = cayley_iters

        self.norm = nn.LayerNorm(n_embd, elementwise_affine=True)

        self.fused_proj = nn.Linear(n_embd, 3 * n_streams * n_streams,
                                    bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize for near-identity behavior at start."""
        with torch.no_grad():
            nn.init.zeros_(self.fused_proj.weight)
            bias = self.fused_proj.bias
            n = self.n_streams
            bias[:n * n].zero_()
            bias[n * n:2 * n * n].zero_()
            bias[2 * n * n:].zero_()

    def compute_mappings(self, x_streams):
        """
        Compute per-token dynamic mixing matrices (Eq. 12/35).

        Args:
            x_streams: (B, T, n, d) multi-stream input

        Returns:
            H_pre:  (B, T, n, n) row-stochastic
            H_post: (B, T, n, n) column-stochastic
            H_res:  (B, T, n, n) approximately orthogonal (iterative Cayley)
        """
        B, T, n, d = x_streams.shape
        x_flat = x_streams.reshape(B, T, -1)
        x_norm = self.norm(x_flat)

        fused = self.fused_proj(x_norm)
        fused = fused.view(B, T, 3, n, n)

        H_pre_raw = fused[:, :, 0]
        H_post_raw = fused[:, :, 1]
        H_res_raw = fused[:, :, 2]

        H_pre = F.softmax(H_pre_raw, dim=-1)
        H_post = F.softmax(H_post_raw, dim=-2)
        H_res = iterative_cayley(H_res_raw, self.cayley_alpha,
                                 self.cayley_iters)

        return H_pre, H_post, H_res

    def forward(self, x_streams, layer_fn, ln):
        """
        Apply JPmHC parallel routing (their Equation 14).

        Per the paper, F is evaluated on a single d-dimensional
        stream-averaged vector, not the full nd-dimensional embedding.

        Args:
            x_streams: (B, T, n, d)
            layer_fn: Attention or MLP function (operates on d dimensions)
            ln: LayerNorm to apply before layer_fn (d-dimensional)

        Returns:
            x_out: (B, T, n, d)
        """
        B, T, n, d = x_streams.shape

        H_pre, H_post, H_res = self.compute_mappings(x_streams)

        # Residual path: H_res · x_streams
        x_mixed = torch.einsum('btij,btjd->btid', H_res, x_streams)

        # Compute path: H_pre → stream average → LN → F → broadcast → H_post
        x_pre = torch.einsum('btij,btjd->btid', H_pre, x_streams)
        x_agg = x_pre.mean(dim=2)  # (B, T, d) — single stream average

        x_ln = ln(x_agg)           # LN on d-dimensional vector
        y = layer_fn(x_ln)         # F on d-dimensional vector → (B, T, d)

        # Broadcast: y ⊗ 1_n → (B, T, n, d)
        y_broadcast = y.unsqueeze(2).expand(-1, -1, n, -1)

        # H_post distributes output across streams
        x_post = torch.einsum('btij,btjd->btid', H_post, y_broadcast)

        x_out = x_mixed + x_post
        return x_out


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight,
                            self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Self-attention operating on d_stream dimensions (single stream width)."""

    def __init__(self, config):
        super().__init__()
        self.d_stream = config.d_stream
        self.n_head = config.n_head_stream
        assert self.d_stream % self.n_head == 0

        self.c_attn = nn.Linear(self.d_stream, 3 * self.d_stream,
                                bias=config.bias)
        self.c_proj = nn.Linear(self.d_stream, self.d_stream,
                                bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size))

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
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0,
                                  float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """FFN operating on d_stream dimensions (single stream width)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_stream, 4 * config.d_stream,
                              bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_stream, config.d_stream,
                                bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with JPmHC parallel routing (Figure 2).

    Sub-layer F (attention/MLP) operates on d_stream = n_embd // n_streams
    per the paper's Section 3.2: "evaluated once on a single p-dimensional
    vector". The JPmHC module generates mixing matrices from the full
    nd-dimensional representation.
    """

    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        self.n_embd = config.n_embd
        self.d_stream = config.n_embd // self.n_streams

        assert config.n_embd % self.n_streams == 0

        # Sub-layer LNs operate on d_stream (single stream, per paper)
        self.ln_1 = LayerNorm(self.d_stream, bias=config.bias)
        self.ln_2 = LayerNorm(self.d_stream, bias=config.bias)

        # Compute n_head for stream-level attention
        n_head_stream = min(getattr(config, 'n_head', 4), self.d_stream)
        while self.d_stream % n_head_stream != 0:
            n_head_stream -= 1

        # Attention and MLP operate on d_stream (per paper Section 3.2)
        stream_cfg = _StreamConfig(
            d_stream=self.d_stream,
            n_head_stream=n_head_stream,
            dropout=config.dropout,
            bias=config.bias,
            block_size=config.block_size,
        )
        self.attn = CausalSelfAttention(stream_cfg)
        self.mlp = MLP(stream_cfg)

        cayley_alpha = getattr(config, 'cayley_alpha', 0.1)
        cayley_iters = getattr(config, 'cayley_iters', 2)

        # JPmHC modules use full n_embd for mixing matrix generation
        self.jpmhc_attn = JPmHCModule(
            n_streams=self.n_streams,
            n_embd=self.n_embd,
            cayley_alpha=cayley_alpha,
            cayley_iters=cayley_iters,
        )
        self.jpmhc_mlp = JPmHCModule(
            n_streams=self.n_streams,
            n_embd=self.n_embd,
            cayley_alpha=cayley_alpha,
            cayley_iters=cayley_iters,
        )

    def forward(self, x):
        B, T, D = x.shape
        x_streams = x.reshape(B, T, self.n_streams, self.d_stream)

        x_streams = self.jpmhc_attn(x_streams, self.attn, self.ln_1)
        x_streams = self.jpmhc_mlp(x_streams, self.mlp, self.ln_2)

        return x_streams.reshape(B, T, D)


@dataclass
class _StreamConfig:
    """Config for stream-level sub-layers (attention, MLP)."""
    d_stream: int = 32
    n_head_stream: int = 4
    dropout: float = 0.0
    bias: bool = True
    block_size: int = 1024


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
    cayley_alpha: float = 0.1
    cayley_iters: int = 2


class GPT(nn.Module):
    """
    GPT with JPmHC (JP Morgan Hyper-Connections with Cayley Retraction).

    Reference: Sengupta, Wang & Brunswic, arXiv:2602.18308

    Faithful to the paper's architecture:
    1. Per-token dynamic generation of all mixing matrices (Eq. 12)
    2. Iterative Cayley retraction with Y₀ = I + αW init (Algorithm 8)
    3. Parallel routing: H_res path || H_pre → F → H_post path (Eq. 14)
    4. F operates on single d-dimensional stream average (Section 3.2)
    5. Row-stochastic H_pre, column-stochastic H_post (Eq. 13)

    Limitations vs E∆-MHC-Geo:
    - Approximate orthogonality (||Y^T Y - I|| < 10^{-3})
    - SO(n) only (no reflection, no eigenvalue -1)
    - Parallel routing (not series pre-transformation)
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
        print(f"JPmHC model - parameters: {self.get_num_params() / 1e6:.2f}M")
        print(f"  n_streams: {config.n_streams}, d_stream: {d_stream}")
        print(f"  Cayley iterations: {config.cayley_iters}, alpha: {config.cayley_alpha}")
        print(f"  Cayley init: Y₀ = I + αW (Algorithm 8)")
        print(f"  Sub-layer F width: {d_stream} (single stream, per paper Sec 3.2)")
        print(f"  Routing: PARALLEL (H_res || H_pre→F→H_post)")

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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
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
