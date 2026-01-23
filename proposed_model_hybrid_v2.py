"""
E∆-Hybrid V2: Enhanced Hybrid with Entropy-Aware Gating

Key improvement over V1:
- Gate now considers BOTH input content AND entropy (Φ)
- gate = σ(W·x + w_φ·Φ + b)

This allows the model to learn:
- "Uncertain + correction phrase → prefer reflection"
- "Uncertain + no correction → prefer rotation"  
- "Confident → preserve (weak operations)"

Mathematical formulation:
- Rotation β: softplus(w_α·Φ + b_rot)  [entropy-driven]
- Reflection β: 2·σ(w_ref·Φ + b_ref)   [NOW also entropy-driven!]
- Gate γ: σ(W·x + w_φ·Φ + b_gate)      [input + entropy]

Design rationale:
- Rotation: Strong when uncertain (explore manifold)
- Reflection: Strong when confident-but-wrong (correction needed)
- Gate: Learns WHEN each is appropriate given context + uncertainty
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeodesicDeltaHybridV2(nn.Module):
    """
    Enhanced Hybrid Operator with Entropy-Aware Gating
    
    Three entropy-influenced components:
    1. Rotation magnitude (β_rot) - entropy-scaled
    2. Reflection magnitude (β_ref) - NOW entropy-aware (optional)
    3. Gate (γ) - input + entropy combined
    
    Args:
        d_model: Model dimension
        n_streams: Number of streams for hyper-connection
        init_gate_bias: Initial bias for gate (0=equal, >0=prefer rotation)
        entropy_gate: Whether gate includes entropy term
        entropy_reflection: Whether reflection β is entropy-aware
    """
    def __init__(self, d_model, n_streams=4, init_gate_bias=0.0,
                 entropy_gate=True, entropy_reflection=False):
        super().__init__()
        assert d_model % n_streams == 0
        
        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        self.entropy_gate = entropy_gate
        self.entropy_reflection = entropy_reflection
        
        # === CAYLEY ROTATION PARAMS ===
        self.u = nn.Parameter(torch.randn(n_streams, 1) * 0.1)
        self.v = nn.Parameter(torch.randn(n_streams, 1) * 0.1)
        
        # Rotation magnitude: β_rot = softplus(w_α·Φ + b_rot)
        self.rot_w_alpha = nn.Parameter(torch.ones(1))
        self.rot_b_init = nn.Parameter(torch.tensor(0.0))
        
        # === HOUSEHOLDER REFLECTION PARAMS ===
        self.k_raw = nn.Parameter(torch.randn(n_streams) * 0.1)
        
        if entropy_reflection:
            # Entropy-aware reflection: β_ref = 2·σ(w_ref·Φ + b_ref)
            self.ref_w_alpha = nn.Parameter(torch.zeros(1))  # Start at 0 (no entropy influence)
            self.ref_b_init = nn.Parameter(torch.tensor(1.0))  # σ(1)≈0.73 → β≈1.46
        else:
            # Static learned reflection magnitude
            self.ref_scale = nn.Parameter(torch.tensor(1.0))
        
        # === ENTROPY-AWARE GATE ===
        # gate = σ(W·x + w_φ·Φ + b)
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_gate_bias)
        
        if entropy_gate:
            # Learnable weight for entropy contribution to gate
            self.gate_w_phi = nn.Parameter(torch.zeros(1))  # Start neutral
        
        # Cache identity matrix
        self.register_buffer('I', torch.eye(n_streams))
    
    def get_purity_proxy(self, x_streams):
        """
        Frobenius Purity Proxy (Linear Entropy)
        
        Φ = 1 - ||G||_F² / (Tr(G))²
        
        where G = X·X^T is the Gram matrix.
        
        Properties:
        - Φ = 0 for pure state (rank-1, confident)
        - Φ → 1-1/n for maximally mixed state (uncertain)
        """
        G = torch.matmul(x_streams, x_streams.transpose(-1, -2))
        frob_sq = torch.sum(G**2, dim=(-1, -2), keepdim=True)
        tr = torch.einsum('...ii->...', G).unsqueeze(-1).unsqueeze(-1)
        tr_sq = tr ** 2 + 1e-6
        phi = 1.0 - (frob_sq / tr_sq)
        return phi  # (B, S, 1, 1)
    
    def cayley_rotation(self, x_streams, beta):
        """
        Apply Cayley rotation with thermodynamic magnitude
        
        Q = (I + βA/2)^{-1}(I - βA/2)
        
        where A = uv^T - vu^T is skew-symmetric.
        """
        A = self.u @ self.v.T - self.v @ self.u.T
        A_scaled = beta.mean() * A
        M = A_scaled / 2
        Q = torch.linalg.solve(self.I + M, self.I - M)
        x_rotated = torch.einsum('ij,bsjd->bsid', Q, x_streams)
        return x_rotated
    
    def householder_reflection(self, x_streams, beta_ref):
        """
        Apply Householder reflection along learned direction
        
        H = I - β·k·k^T
        x' = x - β·(x·k)·k
        """
        k = F.normalize(self.k_raw, dim=0)
        k_expanded = k.view(1, 1, self.n_streams, 1)
        dot = (x_streams * k_expanded).sum(dim=2, keepdim=True)
        x_reflected = x_streams - beta_ref * dot * k_expanded
        return x_reflected
    
    def forward(self, x, return_debug=False):
        B, S, D = x.shape
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        
        # === COMPUTE ENTROPY (Purity Proxy) ===
        phi = self.get_purity_proxy(x_streams)  # (B, S, 1, 1)
        phi_scalar = phi.mean(dim=(1, 2, 3))  # (B,) for gate
        
        # === ENTROPY-AWARE GATE ===
        x_pooled = x.mean(dim=1)  # (B, D)
        gate_logit = self.gate_proj(x_pooled).squeeze(-1)  # (B,)
        
        if self.entropy_gate:
            # Add entropy contribution: high Φ → shift gate
            gate_logit = gate_logit + self.gate_w_phi * phi_scalar
        
        gate = torch.sigmoid(gate_logit)  # (B,)
        gate = gate.view(B, 1, 1, 1)  # Broadcast shape
        
        # === ROTATION: Entropy-driven magnitude ===
        beta_rot = F.softplus(self.rot_w_alpha * phi + self.rot_b_init)
        x_rotated = self.cayley_rotation(x_streams, beta_rot)
        
        # === REFLECTION: Optionally entropy-aware ===
        if self.entropy_reflection:
            # Entropy-aware: β_ref = 2·σ(w·Φ + b)
            phi_mean = phi.mean()  # Scalar for reflection
            beta_ref = 2.0 * torch.sigmoid(self.ref_w_alpha * phi_mean + self.ref_b_init)
        else:
            # Static learned
            beta_ref = torch.sigmoid(self.ref_scale) * 2.0
        
        x_reflected = self.householder_reflection(x_streams, beta_ref)
        
        # === HYBRID OUTPUT ===
        x_hybrid = gate * x_rotated + (1 - gate) * x_reflected
        x_out = x_hybrid.reshape(B, S, D)
        
        if return_debug:
            return x_out, {
                'gate': gate.mean().item(),
                'phi': phi.mean().item(),
                'beta_rot': beta_rot.mean().item(),
                'beta_ref': beta_ref.item() if isinstance(beta_ref, torch.Tensor) else beta_ref,
                'gate_w_phi': self.gate_w_phi.item() if self.entropy_gate else 0.0,
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
    Hybrid V2 Block with entropy-aware gating
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
        # Hybrid V2 operators
        init_gate_bias = getattr(config, 'init_gate_bias', 0.0)
        entropy_gate = getattr(config, 'entropy_gate', True)
        entropy_reflection = getattr(config, 'entropy_reflection', False)
        
        self.hybrid_attn = GeodesicDeltaHybridV2(
            config.n_embd, n_streams=4,
            init_gate_bias=init_gate_bias,
            entropy_gate=entropy_gate,
            entropy_reflection=entropy_reflection
        )
        self.hybrid_mlp = GeodesicDeltaHybridV2(
            config.n_embd, n_streams=4,
            init_gate_bias=init_gate_bias,
            entropy_gate=entropy_gate,
            entropy_reflection=entropy_reflection
        )

    def forward(self, x):
        x_hybrid = self.hybrid_attn(x)
        x = x_hybrid + self.attn(self.ln_1(x_hybrid))
        
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
    # Hybrid V2 specific
    init_gate_bias: float = 0.0      # 0=equal, >0=prefer rotation
    entropy_gate: bool = True         # Gate includes entropy term
    entropy_reflection: bool = False  # Reflection β is entropy-aware


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

        print(f"E∆-Hybrid V2 model - number of parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"  init_gate_bias: {config.init_gate_bias}")
        print(f"  entropy_gate: {config.entropy_gate}")
        print(f"  entropy_reflection: {config.entropy_reflection}")
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
    
    @torch.no_grad()
    def analyze_hybrid_params(self):
        """Analyze learned hybrid parameters across all layers"""
        print("\n=== E∆-Hybrid V2 Parameter Analysis ===\n")
        
        for i, block in enumerate(self.transformer.h):
            print(f"Layer {i}:")
            
            for name, hybrid in [("Attn", block.hybrid_attn), ("MLP", block.hybrid_mlp)]:
                # Gate parameters
                gate_bias = hybrid.gate_proj.bias.item()
                gate_w_phi = hybrid.gate_w_phi.item() if hybrid.entropy_gate else 0.0
                
                # Rotation parameters
                rot_w = hybrid.rot_w_alpha.item()
                rot_b = hybrid.rot_b_init.item()
                A_norm = torch.norm(hybrid.u @ hybrid.v.T - hybrid.v @ hybrid.u.T).item()
                
                # Reflection parameters
                if hybrid.entropy_reflection:
                    ref_w = hybrid.ref_w_alpha.item()
                    ref_b = hybrid.ref_b_init.item()
                    beta_ref_at_phi0 = 2 * torch.sigmoid(torch.tensor(ref_b)).item()
                else:
                    beta_ref = torch.sigmoid(hybrid.ref_scale).item() * 2
                
                print(f"  {name}:")
                print(f"    Gate: bias={gate_bias:.4f}, w_φ={gate_w_phi:.4f}")
                print(f"    Rotation: ||A||={A_norm:.6f}, w_α={rot_w:.4f}, b={rot_b:.4f}")
                if hybrid.entropy_reflection:
                    print(f"    Reflection: w_α={ref_w:.4f}, b={ref_b:.4f}, β(Φ=0)={beta_ref_at_phi0:.4f}")
                else:
                    print(f"    Reflection: β={beta_ref:.4f}")
            print()
