"""
JPmHC Implementation (JP Morgan Hyper-Connections with Cayley Retraction)
Reference: Sengupta, Wang & Brunswic, arXiv:2602.18308, February 2026

This implements JPmHC's key architectural choices for fair comparison:

Architecture (Parallel Routing, their Equation 14):
    x_out = H_res · x_streams + H_post · (y ⊗ 1_n)
    where y = F(avg(H_pre · x_streams))

The residual path (H_res) and compute path (H_pre → F → H_post)
operate in PARALLEL on the same input, applying different transforms.

Coefficient Computation (their Equation 12):
    [H̃_pre | H̃_post | H̃_res] = W_fused · LayerNorm(x_flat)

Constraints:
    H_pre  = softmax(H̃_pre, dim=-1)        → row-stochastic
    H_post = softmax(H̃_post, dim=-2)       → column-stochastic
    H_res  = IterativeCayley(H̃_res - H̃_res^T)  → approx orthogonal

Iterative Cayley (their Algorithm 3, Appendix B.2):
    W = H̃_res - H̃_res^T  (skew-symmetrize)
    Y_0 = I
    Y_{i+1} = I + (α/2) W (I + Y_i),  i = 0, ..., s-1
    with α = 0.1, s = 2 iterations

Key differences from E∆-MHC-Geo:
    - Parallel routing (vs series pre-transformation)
    - Full-rank Cayley generator (vs rank-2)
    - Iterative approximation (vs exact analytical solve)
    - SO(n) only (vs full O(n) with Householder gate)
    - Per-token dynamic pre/post (vs learned weight matrices)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def iterative_cayley(H_tilde, alpha=0.1, n_iters=2):
    """
    Iterative fixed-point Cayley retraction (JPmHC Algorithm 3).

    Approximates Q = (I + W/2)^{-1}(I - W/2) via fixed-point iteration:
        Y_{i+1} = I + (α/2) W (I + Y_i)

    The result is approximately orthogonal with deviation
    ||Y^T Y - I||_max < 10^{-3} at s=2, α=0.1 (their Proposition I.1).

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

    Y = I.expand_as(W).clone()
    for _ in range(n_iters):
        Y = I + (alpha / 2) * torch.matmul(W, I + Y)

    return Y


class JPmHCModule(nn.Module):
    """
    JPmHC dynamic routing module with per-token matrix generation.

    Generates all three n×n mixing matrices (H_pre, H_post, H_res)
    dynamically per token via a single fused linear projection, then
    applies parallel routing: x_out = H_res·x + H_post·F(avg(H_pre·x)).
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

        # Fused projection: predict 3 n×n matrices in one shot
        self.fused_proj = nn.Linear(n_embd, 3 * n_streams * n_streams,
                                    bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize for near-identity behavior at start."""
        with torch.no_grad():
            nn.init.zeros_(self.fused_proj.weight)
            bias = self.fused_proj.bias
            n = self.n_streams
            # H_pre bias → identity-like (uniform softmax)
            bias[:n * n].zero_()
            # H_post bias → identity-like (uniform softmax)
            bias[n * n:2 * n * n].zero_()
            # H_res bias → identity (Cayley of zero = I)
            bias[2 * n * n:].zero_()

    def compute_mappings(self, x_streams):
        """
        Compute per-token dynamic mixing matrices.

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
        Apply JPmHC parallel routing (their Equation 14):
            x_out = H_res · x_streams + H_post · (F(avg(H_pre · x)) ⊗ 1_n)

        Args:
            x_streams: (B, T, n, d)
            layer_fn: Attention or MLP function
            ln: LayerNorm to apply before layer_fn

        Returns:
            x_out: (B, T, n, d)
        """
        B, T, n, d = x_streams.shape

        H_pre, H_post, H_res = self.compute_mappings(x_streams)

        # Residual path: H_res · x_streams
        x_mixed = torch.einsum('btij,btjd->btid', H_res, x_streams)

        # Compute path: H_pre → average → F → H_post broadcast
        x_pre = torch.einsum('btij,btjd->btid', H_pre, x_streams)
        x_agg = x_pre.mean(dim=2)  # (B, T, d)

        x_agg_full = x_agg.unsqueeze(2).expand(-1, -1, n, -1)
        x_agg_full = x_agg_full.reshape(B, T, -1)
        x_ln = ln(x_agg_full)
        f_out = layer_fn(x_ln)
        f_out_streams = f_out.reshape(B, T, n, d)

        x_broadcast = torch.einsum('btij,btjd->btid', H_post,
                                   f_out_streams)

        x_out = x_mixed + x_broadcast
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
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd,
                                bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,
                                bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
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
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0,
                                  float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd,
                              bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd,
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
    Transformer block with JPmHC parallel routing.

    Uses per-token dynamic generation of H_pre, H_post, H_res
    with iterative Cayley retraction for H_res.
    """

    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        self.n_embd = config.n_embd
        self.d_stream = config.n_embd // self.n_streams

        assert config.n_embd % self.n_streams == 0

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

        cayley_alpha = getattr(config, 'cayley_alpha', 0.1)
        cayley_iters = getattr(config, 'cayley_iters', 2)

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

    Key features:
    1. Per-token dynamic generation of all mixing matrices
    2. Iterative Cayley retraction for approximate orthogonality
    3. Parallel routing: H_res path || H_pre → F → H_post path
    4. Row-stochastic H_pre, column-stochastic H_post

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

        print(f"JPmHC model - parameters: {self.get_num_params() / 1e6:.2f}M")
        print(f"  n_streams: {config.n_streams}")
        print(f"  Cayley iterations: {config.cayley_iters}")
        print(f"  Cayley alpha: {config.cayley_alpha}")
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
