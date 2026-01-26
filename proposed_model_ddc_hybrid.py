"""
Data-Dependent Cayley Hybrid (DDC-Hybrid) Transformer

Combines:
1. Data-Dependent Cayley (DDC) - Input-adaptive rotation with guaranteed orthogonality
2. Householder Reflection (at β=2) - Can negate information (eigenvalue -1)
3. Learned Gate - Selects between rotation and reflection based on input

This achieves the best of both worlds:
- Rotation for geometric tasks (isometric, smooth)
- Reflection for correction tasks (can negate)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    n_streams: int = 4  # Number of streams for rotation


class DDCHybridOperator(nn.Module):
    """
    Data-Dependent Cayley Hybrid Operator
    
    Combines:
    - DDC: x' = Q(x) @ x where Q(x) ∈ SO(n) is input-dependent rotation
    - Householder: x' = (I - 2kk^T) @ x where k(x) is input-dependent direction
    - Gate γ: (0,1) selects between rotation (γ→1) and reflection (γ→0)
    - Residual gate α: (0,1) controls identity vs transformation
    
    Final output: α * [γ * DDC(x) + (1-γ) * Householder(x)] + (1-α) * x
    
    Special cases:
    - α → 0: Classical residual (identity) 
    - α → 1, γ → 1: Full DDC rotation
    - α → 1, γ → 0: Full Householder reflection
    """
    
    def __init__(self, d_model: int, n_streams: int = 4):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        self.d_model = d_model
        
        hidden_dim = d_model // 4
        
        # ===== DDC Component (Data-Dependent Cayley) =====
        # Generator networks: x → u(x), v(x)
        self.u_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_streams)
        )
        
        self.v_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_streams)
        )
        
        # Beta network for rotation magnitude
        self.beta_rot_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # ===== Householder Component =====
        # k network: x → k(x), the reflection direction
        self.k_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
        # β_ref is fixed at 2 for orthogonality
        
        # ===== Gate Component =====
        # Learns to select rotation vs reflection based on input
        self.gate_net = nn.Linear(d_model, 1)
        
        # Identity buffer for Cayley
        self.register_buffer('I_streams', torch.eye(n_streams))
        
        # Initialize for balanced initial behavior
        self._init_weights()
    
    def _init_weights(self):
        # Small initialization for Cayley generators
        for net in [self.u_net, self.v_net]:
            if hasattr(net[-1], 'weight'):
                nn.init.normal_(net[-1].weight, std=0.01)
                nn.init.zeros_(net[-1].bias)
        
        # Initialize gate to 0.5 (balanced)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)
    
    def cayley_transform(self, A: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Cayley transform: Q = (I + βA/2)^{-1} (I - βA/2)
        
        GUARANTEED to be orthogonal for ANY A that is skew-symmetric.
        """
        B = A.shape[0]
        M = (beta / 2) * A  # (B, n, n)
        
        I = self.I_streams.unsqueeze(0).expand(B, -1, -1)
        I_plus_M = I + M
        I_minus_M = I - M
        
        Q = torch.linalg.solve(I_plus_M, I_minus_M)
        return Q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DDC-Hybrid operator.
        
        Returns: γ * rotation(x) + (1-γ) * reflection(x)
        """
        B, S, D = x.shape
        
        # Pool for computing operators
        x_pooled = x.mean(dim=1)  # (B, D)
        
        # ===== DDC ROTATION =====
        u = self.u_net(x_pooled)  # (B, n_streams)
        v = self.v_net(x_pooled)  # (B, n_streams)
        beta_rot = self.beta_rot_net(x_pooled).unsqueeze(-1)  # (B, 1, 1)
        
        # Skew-symmetric A = uv^T - vu^T (ALWAYS skew-symmetric)
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        
        # Cayley rotation matrix (ALWAYS orthogonal)
        Q = self.cayley_transform(A, beta_rot)  # (B, n_streams, n_streams)
        
        # Apply rotation in stream space
        x_streams = x.view(B, S, self.n_streams, self.d_stream)
        x_rotated = torch.einsum('bij,bsjd->bsid', Q, x_streams)
        x_rotated = x_rotated.reshape(B, S, D)
        
        # ===== HOUSEHOLDER REFLECTION =====
        k = self.k_net(x_pooled)  # (B, D)
        k = F.normalize(k, dim=-1)  # Unit vector
        k = k.unsqueeze(1)  # (B, 1, D)
        
        # H = I - 2kk^T, so H @ x = x - 2k(k^T x)
        # At β=2, this is guaranteed orthogonal with eigenvalue -1 along k
        dot = (x * k).sum(dim=-1, keepdim=True)  # (B, S, 1)
        x_reflected = x - 2 * dot * k  # (B, S, D)
        
        # ===== GATE =====
        gamma = torch.sigmoid(self.gate_net(x_pooled))  # (B, 1)
        gamma = gamma.unsqueeze(1)  # (B, 1, 1)
        
        # ===== HYBRID OUTPUT =====
        # γ → 1: rotation dominates (geometric mode)
        # γ → 0: reflection dominates (correction mode)
        x_hybrid = gamma * x_rotated + (1 - gamma) * x_reflected
        
        return x_hybrid


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
        self.flash = hasattr(F, 'scaled_dot_product_attention')
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
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                               dropout_p=self.dropout if self.training else 0,
                                               is_causal=True)
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


class DDCHybridBlock(nn.Module):
    """
    DDC-Hybrid Block with mHC-style Pre/Post Mappings
    
    Implements the full mHC framework with hybrid geometric operator:
    X_{l+1} = G_γ(X_l)X_l + H_post^T @ F(H_pre @ LN(G_γ(X_l)X_l))
    
    Where:
    - G_γ(X) = γ * DDC(X) + (1-γ) * Householder(X) is the hybrid operator
    - H_pre aggregates streams before attention/MLP (from mHC paper)
    - H_post broadcasts output back to streams (from mHC paper)
    
    This unifies:
    - mHC: multi-stream residual with pre/post mappings
    - DDL: input-adaptive transformations + negation capability
    - Cayley: unconditional orthogonality guarantees
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
        # DDC-Hybrid operators (DDC rotation + Householder reflection + learned gate)
        self.hybrid_attn = DDCHybridOperator(config.n_embd, self.n_streams)
        self.hybrid_mlp = DDCHybridOperator(config.n_embd, self.n_streams)
        
        # === mHC Pre/Post Mappings (from DeepSeek mHC paper arXiv:2512.24880) ===
        # H_pre: aggregates n streams → prepares input for layer function
        # H_post: broadcasts layer output → distributes to n streams
        
        # For Attention block
        self.h_pre_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # For MLP block
        self.h_pre_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Initialize pre/post mappings to identity (stable starting point)
        with torch.no_grad():
            self.h_pre_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_post_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_pre_mlp.weight.copy_(torch.eye(config.n_embd))
            self.h_post_mlp.weight.copy_(torch.eye(config.n_embd))
    
    def forward(self, x):
        # === ATTENTION BLOCK ===
        # Step 1: Apply Hybrid operator (DDC + Householder with learned gate)
        x_transformed = self.hybrid_attn(x)
        
        # Step 2: H_pre → Attention → H_post (mHC-style)
        x_normed = self.ln_1(x_transformed)
        x_pre = self.h_pre_attn(x_normed)          # H_pre: aggregate/project
        attn_out = self.attn(x_pre)                 # F: attention function
        x_post = self.h_post_attn(attn_out)        # H_post: broadcast/project back
        
        # Step 3: Residual connection (with transformed input)
        x = x_transformed + x_post
        
        # === MLP BLOCK ===
        x_transformed = self.hybrid_mlp(x)
        x_normed = self.ln_2(x_transformed)
        x_pre = self.h_pre_mlp(x_normed)
        mlp_out = self.mlp(x_pre)
        x_post = self.h_post_mlp(mlp_out)
        x = x_transformed + x_post
        
        return x


class DDCHybridTransformer(nn.Module):
    """
    Data-Dependent Cayley Hybrid Transformer
    
    Combines:
    - Data-Dependent Cayley for input-adaptive rotation with guaranteed orthogonality
    - Householder reflection for negation capability
    - Learned gate for adaptive selection
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
            h = nn.ModuleList([DDCHybridBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"DDC-Hybrid Transformer: {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
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
        """Extract gate values from all layers for analysis."""
        gates = []
        for i, block in enumerate(self.transformer.h):
            with torch.no_grad():
                # Can't get actual values without input, return learned biases as proxy
                gates.append({
                    'layer': i,
                    'attn_gate_bias': block.hybrid_attn.gate_net.bias.item(),
                    'mlp_gate_bias': block.hybrid_mlp.gate_net.bias.item(),
                })
        return gates
    
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


# Alias for compatibility
GPT = DDCHybridTransformer
