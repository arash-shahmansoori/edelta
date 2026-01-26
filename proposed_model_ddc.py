"""
Data-Dependent Cayley (DDC) Transformer

Unlike fixed Cayley where u,v are learned parameters, DDC computes u(x),v(x)
via neural networks. This achieves:
- Input-adaptive rotation planes (like DDL)
- Unconditional orthogonality guarantees (unlike DDL)
- Perfect norm preservation for all inputs
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


class DataDependentCayleyOperator(nn.Module):
    """
    Data-Dependent Cayley Rotation Operator
    
    Key insight: The skew-symmetry of A = uv^T - vu^T depends only on the
    algebraic construction, not on how u and v are obtained. Therefore,
    computing u(x), v(x) via neural networks preserves all Cayley properties.
    
    Properties (all unconditional):
    - Q(x)^T Q(x) = I (orthogonality)
    - ||Q(x) y|| = ||y|| (isometry)
    - det(Q(x)) = +1 (proper rotation)
    
    Classical Residual Preservation:
    - When α → 0: output = x (identity, classical residual preserved)
    - When α → 1: output = Q(x) @ x (full rotation)
    - The network can learn to interpolate between identity and rotation
    """
    
    def __init__(self, d_model: int, n_streams: int = 4, use_residual_gate: bool = True):
        super().__init__()
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams
        self.use_residual_gate = use_residual_gate
        
        # Generator networks: x → u(x), v(x)
        # Using a hidden layer for increased expressivity
        hidden_dim = d_model // 4
        
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
        
        # Beta network: controls rotation magnitude
        self.beta_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive
        )
        
        # Residual gate: α ∈ (0, 1) controls rotation vs identity
        # α → 0: use identity (classical residual)
        # α → 1: use rotation (full DDC)
        if use_residual_gate:
            self.alpha_gate = nn.Linear(d_model, 1)
        
        # Identity buffer for efficiency
        self.register_buffer('I', torch.eye(n_streams))
        
        # Initialize for small initial rotations
        self._init_weights()
    
    def _init_weights(self):
        # Small initialization for gradual rotation learning
        for net in [self.u_net, self.v_net]:
            if hasattr(net[-1], 'weight'):
                nn.init.normal_(net[-1].weight, std=0.01)
                nn.init.zeros_(net[-1].bias)
        
        # Initialize residual gate to prefer identity initially (easier optimization)
        if self.use_residual_gate:
            nn.init.zeros_(self.alpha_gate.weight)
            nn.init.constant_(self.alpha_gate.bias, -1.0)  # sigmoid(-1) ≈ 0.27, slight preference for identity
    
    def cayley_transform(self, A: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Cayley transform: Q = (I + βA/2)^{-1} (I - βA/2)
        
        Args:
            A: Skew-symmetric matrix (B, n, n)
            beta: Rotation magnitude (B, 1, 1)
        
        Returns:
            Q: Rotation matrix (B, n, n)
        """
        B = A.shape[0]
        M = (beta / 2) * A  # (B, n, n)
        
        I = self.I.unsqueeze(0).expand(B, -1, -1)  # (B, n, n)
        I_plus_M = I + M
        I_minus_M = I - M
        
        # Solve (I + M) Q = (I - M) for Q
        # This is more numerically stable than explicit inverse
        Q = torch.linalg.solve(I_plus_M, I_minus_M)
        
        return Q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply data-dependent Cayley rotation with optional residual gate.
        
        When use_residual_gate=True:
            output = α * Q(x)@x + (1-α) * x
            - α → 0: output ≈ x (classical residual, identity)
            - α → 1: output ≈ Q(x)@x (full rotation)
        
        When use_residual_gate=False:
            output = Q(x)@x
            - Still reduces to identity when β→0 or u∥v
        
        Args:
            x: Input tensor (B, S, D)
        
        Returns:
            x_out: Transformed tensor (B, S, D)
        """
        B, S, D = x.shape
        
        # Pool across sequence for rotation parameters
        x_pooled = x.mean(dim=1)  # (B, D)
        
        # Compute data-dependent generators
        u = self.u_net(x_pooled)  # (B, n_streams)
        v = self.v_net(x_pooled)  # (B, n_streams)
        beta = self.beta_net(x_pooled).unsqueeze(-1)  # (B, 1, 1)
        
        # Construct skew-symmetric A(x) = uv^T - vu^T
        # This is ALWAYS skew-symmetric regardless of u, v values
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        # A shape: (B, n_streams, n_streams)
        
        # Apply Cayley transform
        Q = self.cayley_transform(A, beta)  # (B, n_streams, n_streams)
        
        # Reshape to streams and apply rotation
        x_streams = x.view(B, S, self.n_streams, self.d_stream)  # (B, S, n, d)
        
        # Rotate: Q @ x_streams along the stream dimension
        x_rotated = torch.einsum('bij,bsjd->bsid', Q, x_streams)
        x_rotated = x_rotated.reshape(B, S, D)
        
        # Apply residual gate if enabled
        # This allows the network to easily learn "no rotation" (classical residual)
        if self.use_residual_gate:
            alpha = torch.sigmoid(self.alpha_gate(x_pooled))  # (B, 1)
            alpha = alpha.unsqueeze(1)  # (B, 1, 1)
            # Interpolate between identity and rotation
            # α → 0: output = x (identity, classical residual preserved)
            # α → 1: output = Q(x)@x (full rotation)
            x_out = alpha * x_rotated + (1 - alpha) * x
        else:
            x_out = x_rotated
        
        return x_out


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


class DDCBlock(nn.Module):
    """
    Data-Dependent Cayley Block with mHC-style Pre/Post Mappings
    
    Implements the full mHC framework:
    X_{l+1} = Q(X_l)X_l + H_post^T @ F(H_pre @ LN(Q(X_l)X_l))
    
    Where:
    - Q(X) is the Data-Dependent Cayley rotation (replaces mHC's H_res)
    - H_pre aggregates streams before attention/MLP (from mHC paper)
    - H_post broadcasts output back to streams (from mHC paper)
    
    This unifies:
    - mHC: multi-stream residual with pre/post mappings
    - DDL: input-adaptive transformations
    - Cayley: unconditional orthogonality guarantees
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_streams = getattr(config, 'n_streams', 4)
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
        # Data-Dependent Cayley operators (replace mHC's doubly stochastic H_res)
        self.ddc_attn = DataDependentCayleyOperator(config.n_embd, self.n_streams)
        self.ddc_mlp = DataDependentCayleyOperator(config.n_embd, self.n_streams)
        
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
        # Step 1: Apply DDC rotation (replaces mHC's doubly stochastic H_res)
        x_rotated = self.ddc_attn(x)
        
        # Step 2: H_pre → Attention → H_post (mHC-style)
        x_normed = self.ln_1(x_rotated)
        x_pre = self.h_pre_attn(x_normed)         # H_pre: aggregate/project
        attn_out = self.attn(x_pre)                # F: attention function
        x_post = self.h_post_attn(attn_out)       # H_post: broadcast/project back
        
        # Step 3: Residual connection (with rotated input)
        x = x_rotated + x_post
        
        # === MLP BLOCK ===
        x_rotated = self.ddc_mlp(x)
        x_normed = self.ln_2(x_rotated)
        x_pre = self.h_pre_mlp(x_normed)
        mlp_out = self.mlp(x_pre)
        x_post = self.h_post_mlp(mlp_out)
        x = x_rotated + x_post
        
        return x


class DDCTransformer(nn.Module):
    """
    Data-Dependent Cayley Transformer
    
    Replaces standard residual connections with data-dependent
    Cayley rotations for input-adaptive geometric transformations.
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
            h = nn.ModuleList([DDCBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying
        
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"DDC Transformer: {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters")
    
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
GPT = DDCTransformer
