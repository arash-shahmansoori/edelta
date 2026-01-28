"""
Pure mHC Implementation (DeepSeek's Manifold-Constrained Hyper-Connections)
Reference: arXiv:2512.24880

This implements the REAL mHC with DATA-DEPENDENT Sinkhorn-Knopp projection 
onto the doubly stochastic manifold, following the paper exactly.

Architecture (Paper Equation 3):
    x_{l+1} = H_l^res · x_l + H_l^post^T · F(H_l^pre · x_l, W_l)

Coefficient Computation (Paper Equations 5, 7):
    x̃_l' = RMSNorm(x̃_l)
    H̃_l^pre = α^pre · (x̃_l' · φ^pre) + b^pre
    H̃_l^post = α^post · (x̃_l' · φ^post) + b^post
    H̃_l^res = α^res · mat(x̃_l' · φ^res) + b^res

Final Mappings (Paper Equation 8):
    H_l^pre = σ(H̃_l^pre)
    H_l^post = 2σ(H̃_l^post)
    H_l^res = Sinkhorn-Knopp(H̃_l^res)

Key features:
- Multi-stream residual (n streams, each of dimension d/n)
- DATA-DEPENDENT coefficient computation (not static!)
- Doubly stochastic mixing via Sinkhorn-Knopp algorithm
- Signal energy conservation (rows and columns sum to 1)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """RMSNorm as used in mHC paper."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def sinkhorn_knopp(M, n_iters=20, eps=1e-8):
    """
    Sinkhorn-Knopp projection onto the doubly stochastic manifold.
    
    Paper Equation (9):
        M^(0) = exp(H̃_l^res)
        M^(t) = T_r(T_c(M^(t-1)))
    
    A doubly stochastic matrix has all rows and columns summing to 1,
    ensuring signal energy conservation during mixing.
    """
    # Ensure positive via exponent (paper Eq. 9)
    M = torch.exp(M)
    
    # Alternating row/column normalization
    for _ in range(n_iters):
        # Row normalization T_r
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        # Column normalization T_c
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    
    return M


class DataDependentMHC(nn.Module):
    """
    Data-Dependent mHC Operator following paper Equations (5), (7), (8).
    
    KEY INSIGHT: The mixing coefficients are computed FROM THE INPUT,
    not just static learned parameters!
    
    Paper Equation (7):
        x̃_l' = RMSNorm(vec(x_l))
        H̃_l^pre = α^pre · (x̃_l' · φ^pre) + b^pre
        H̃_l^post = α^post · (x̃_l' · φ^post) + b^post  
        H̃_l^res = α^res · mat(x̃_l' · φ^res) + b^res
    
    Paper Equation (8):
        H_l^pre = σ(H̃_l^pre)
        H_l^post = 2σ(H̃_l^post)
        H_l^res = Sinkhorn-Knopp(H̃_l^res)
    """
    def __init__(self, n_streams=4, d_stream=None, n_embd=None, 
                 n_sinkhorn_iters=20, alpha_init=0.01):
        super().__init__()
        self.n_streams = n_streams
        self.n_sinkhorn_iters = n_sinkhorn_iters
        
        # Total dimension
        if n_embd is not None:
            self.n_embd = n_embd
            self.d_stream = n_embd // n_streams
        else:
            self.d_stream = d_stream
            self.n_embd = n_streams * d_stream
        
        # RMSNorm on flattened input (paper Eq. 7)
        self.rms_norm = RMSNorm(self.n_embd)
        
        # === Dynamic mapping projections (φ) ===
        # These compute INPUT-DEPENDENT coefficients
        # φ^pre: R^{nC} → R^{n}
        self.phi_pre = nn.Linear(self.n_embd, n_streams, bias=False)
        # φ^post: R^{nC} → R^{n}
        self.phi_post = nn.Linear(self.n_embd, n_streams, bias=False)
        # φ^res: R^{nC} → R^{n²}
        self.phi_res = nn.Linear(self.n_embd, n_streams * n_streams, bias=False)
        
        # === Static biases (b) ===
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams, n_streams))
        
        # === Learnable gating factors (α) ===
        # Initialized small per paper Table 5
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        
        # Initialize biases for identity-like behavior
        self._init_biases()
    
    def _init_biases(self):
        """Initialize biases for near-identity behavior at start."""
        with torch.no_grad():
            # H_pre: uniform aggregation → σ(0) = 0.5, so init to give ~1/n
            self.b_pre.fill_(0.0)
            # H_post: uniform broadcast → 2σ(0) = 1.0
            self.b_post.fill_(0.0)
            # H_res: identity mixing → need to init so Sinkhorn gives ~identity
            self.b_res.copy_(torch.eye(self.n_streams) * 2.0)
    
    def compute_mappings(self, x_streams):
        """
        Compute data-dependent mHC mappings.
        
        Args:
            x_streams: (B, T, n_streams, d_stream) multi-stream input
            
        Returns:
            H_pre: (B, T, n_streams) aggregation weights
            H_post: (B, T, n_streams) broadcast weights  
            H_res: (B, T, n_streams, n_streams) doubly stochastic mixing
        """
        B, T, n, d = x_streams.shape
        
        # Flatten to (B, T, nC) for computing mappings
        x_flat = x_streams.reshape(B, T, -1)  # (B, T, n_embd)
        
        # Apply RMSNorm (paper Eq. 7: x̃_l' = RMSNorm(x̃_l))
        x_norm = self.rms_norm(x_flat)  # (B, T, n_embd)
        
        # === Compute dynamic + static mappings (paper Eq. 7) ===
        # H̃ = α · (x̃' · φ) + b
        
        H_tilde_pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre  # (B, T, n)
        H_tilde_post = self.alpha_post * self.phi_post(x_norm) + self.b_post  # (B, T, n)
        H_tilde_res = self.alpha_res * self.phi_res(x_norm).view(B, T, n, n) + self.b_res  # (B, T, n, n)
        
        # === Apply constraints (paper Eq. 8) ===
        H_pre = torch.sigmoid(H_tilde_pre)  # (B, T, n)
        H_post = 2.0 * torch.sigmoid(H_tilde_post)  # (B, T, n)
        H_res = sinkhorn_knopp(H_tilde_res, self.n_sinkhorn_iters)  # (B, T, n, n)
        
        return H_pre, H_post, H_res
    
    def forward(self, x_streams, layer_fn, ln):
        """
        Apply full mHC transition (paper Equation 3):
            x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)
        
        Args:
            x_streams: (B, T, n_streams, d_stream) input
            layer_fn: The layer function F (attention or MLP)
            ln: LayerNorm to apply before layer_fn
            
        Returns:
            x_out: (B, T, n_streams, d_stream) output
        """
        B, T, n, d = x_streams.shape
        
        # Compute data-dependent mappings
        H_pre, H_post, H_res = self.compute_mappings(x_streams)
        
        # === H_res · x_l (mix streams) ===
        # (B,T,n,n) @ (B,T,n,d) -> (B,T,n,d)
        x_mixed = torch.einsum('btij,btjd->btid', H_res, x_streams)
        
        # === H_pre · x_l (aggregate for layer function) ===
        # H_pre: (B, T, n), x_streams: (B, T, n, d)
        # Weighted sum over streams -> (B, T, d)
        x_agg = torch.einsum('btn,btnd->btd', H_pre, x_streams)
        
        # Expand aggregated back to full dimension for layer function
        # The layer function expects (B, T, n_embd)
        x_agg_full = x_agg.unsqueeze(2).expand(-1, -1, n, -1).reshape(B, T, -1)
        
        # Apply layer function F
        x_ln = ln(x_agg_full)
        f_out = layer_fn(x_ln)  # (B, T, n_embd)
        f_out_streams = f_out.reshape(B, T, n, d)  # (B, T, n, d)
        
        # === H_post^T · F(...) (broadcast back) ===
        # H_post: (B, T, n), f_out_streams: (B, T, n, d)
        # Multiply each stream by its weight
        x_broadcast = H_post.unsqueeze(-1) * f_out_streams  # (B, T, n, d)
        
        # === Final: x_{l+1} = H_res · x_l + H_post^T · F(...) ===
        x_out = x_mixed + x_broadcast
        
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
    Transformer block with DATA-DEPENDENT mHC (paper Equations 3, 7, 8).
    
    Implements multi-stream residual with doubly stochastic mixing
    where the mixing coefficients are computed from the input.
    """
    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        self.n_embd = config.n_embd
        self.d_stream = config.n_embd // self.n_streams
        
        assert config.n_embd % self.n_streams == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_streams ({self.n_streams})"
        
        # Layer norms
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        # Core layer functions
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
        # Data-dependent mHC modules
        n_sinkhorn = getattr(config, 'n_sinkhorn_iters', 20)
        alpha_init = getattr(config, 'alpha_init', 0.01)
        
        self.mhc_attn = DataDependentMHC(
            n_streams=self.n_streams,
            n_embd=self.n_embd,
            n_sinkhorn_iters=n_sinkhorn,
            alpha_init=alpha_init
        )
        self.mhc_mlp = DataDependentMHC(
            n_streams=self.n_streams,
            n_embd=self.n_embd,
            n_sinkhorn_iters=n_sinkhorn,
            alpha_init=alpha_init
        )

    def forward(self, x):
        B, T, D = x.shape
        
        # Reshape to streams: (B, T, D) -> (B, T, n_streams, d_stream)
        x_streams = x.reshape(B, T, self.n_streams, self.d_stream)
        
        # Attention sub-layer with mHC
        x_streams = self.mhc_attn(x_streams, self.attn, self.ln_1)
        
        # MLP sub-layer with mHC
        x_streams = self.mhc_mlp(x_streams, self.mlp, self.ln_2)
        
        # Flatten back to (B, T, D)
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
    n_sinkhorn_iters: int = 20
    alpha_init: float = 0.01


class GPT(nn.Module):
    """
    GPT with DATA-DEPENDENT mHC (Manifold-Constrained Hyper-Connections).
    
    Accurate implementation following arXiv:2512.24880.
    
    Key features:
    1. DATA-DEPENDENT mappings computed from input (paper Eq. 7)
    2. Sinkhorn-Knopp projection ensures doubly stochastic H_res
    3. Sigmoid constraints: H_pre = σ(...), H_post = 2σ(...)
    
    Limitations vs E∆-MHC-Geo:
    - Spectral collapse: eigenvalues |λ| ≤ 1, causes oversmoothing
    - Not isometric: ||M·x|| ≤ ||x||, signal energy decreases
    - Iterative: Sinkhorn needs 20 iterations (approximate)
    - No reflection: cannot achieve det=-1 or eigenvalue λ=-1
    """
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

        print(f"Pure mHC model (DATA-DEPENDENT) - parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"  n_streams: {config.n_streams}")
        print(f"  Sinkhorn iterations: {config.n_sinkhorn_iters}")
        print(f"  Alpha init: {config.alpha_init}")
        print(f"  Mappings: DATA-DEPENDENT (per paper Eq. 7)")

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
