"""
Mixer modules for TRM integration: JPmHC and E∆ variants.

These modules wrap the attention and FFN sub-blocks in TRM's
recursive transformer blocks, adding multi-stream residual connections
with orthogonal constraints.

Both mixers follow the parallel routing topology from JPmHC (Eq. 14):
    x_out = H_res · x_streams + H_post · (F(avg(H_pre · x_streams)) ⊗ 1_n)

The key difference:
- JPmHC: H_res via iterative Cayley (SO(n) only, approximate)
- E∆:    H_res via exact Cayley + Householder gate (full O(n), exact)

Usage in TRM Block:
    # Replace: hidden = rms_norm(hidden + self.self_attn(hidden))
    # With:    hidden = self.mixer(hidden, self.self_attn, self.norm_eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JPmHCMixer(nn.Module):
    """
    JPmHC mixer for TRM integration.

    Faithful to arXiv:2602.18308:
    - Fused projection: LN(x_flat) → [H_pre | H_post | H_res]
    - H_pre: row-stochastic (softmax dim=-1)
    - H_post: column-stochastic (softmax dim=-2)
    - H_res: iterative Cayley retraction (Y₀ = I + αW, Algorithm 8)
    - F operates on d_stream = hidden_size // n_streams
    """

    def __init__(self, hidden_size, n_streams=4, cayley_alpha=0.1, cayley_iters=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        self.d_stream = hidden_size // n_streams
        self.cayley_alpha = cayley_alpha
        self.cayley_iters = cayley_iters

        assert hidden_size % n_streams == 0

        self.norm = nn.LayerNorm(hidden_size)
        self.fused_proj = nn.Linear(hidden_size, 3 * n_streams * n_streams, bias=True)

        with torch.no_grad():
            nn.init.zeros_(self.fused_proj.weight)
            self.fused_proj.bias.zero_()

        self.register_buffer('I', torch.eye(n_streams))

    def iterative_cayley(self, H_tilde):
        W = H_tilde - H_tilde.transpose(-1, -2)
        I = self.I
        Y = I + self.cayley_alpha * W
        for _ in range(self.cayley_iters):
            Y = I + (self.cayley_alpha / 2) * torch.matmul(W, I + Y)
        return Y

    def forward(self, x_streams, layer_fn, rms_norm_eps):
        """
        Args:
            x_streams: (B, T, n, d) multi-stream input
            layer_fn: attention or FFN function
            rms_norm_eps: epsilon for RMS normalization
        Returns:
            x_out: (B, T, n, d)
        """
        B, T, n, d = x_streams.shape

        x_flat = x_streams.reshape(B, T, -1)
        x_norm = self.norm(x_flat)
        fused = self.fused_proj(x_norm).view(B, T, 3, n, n)

        H_pre = F.softmax(fused[:, :, 0], dim=-1)
        H_post = F.softmax(fused[:, :, 1], dim=-2)
        H_res = self.iterative_cayley(fused[:, :, 2])

        x_mixed = torch.einsum('btij,btjd->btid', H_res, x_streams)

        x_pre = torch.einsum('btij,btjd->btid', H_pre, x_streams)
        x_agg = x_pre.mean(dim=2)  # (B, T, d)

        y = layer_fn(x_agg)
        y_bc = y.unsqueeze(2).expand(-1, -1, n, -1)
        x_post = torch.einsum('btij,btjd->btid', H_post, y_bc)

        x_out = x_mixed + x_post
        return x_out


class EdeltaMixer(nn.Module):
    """
    E∆ mixer for TRM integration.

    Combines E∆'s full O(n) geometric operator with JPmHC-style routing:
    - Exact Cayley rotation (analytical solve, not iterative)
    - Householder reflection on full dimension (for negation capability)
    - Learned gate γ for automatic operator selection
    - Dynamic H_pre/H_post routing (like JPmHC)
    - F operates on d_stream = hidden_size // n_streams
    """

    def __init__(self, hidden_size, n_streams=4, geo_hidden_dim=48,
                 gate_reg_weight=0.1, init_gate_bias=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        self.d_stream = hidden_size // n_streams
        self.gate_reg_weight = gate_reg_weight

        assert hidden_size % n_streams == 0
        n = n_streams

        # Geometric operator: fused projection for Cayley + Householder + gate
        # Outputs: u(n) + v(n) + β(1) + k(d) + γ(1) = 2n + 2 + d
        self.geo_out_dim = 2 * n + 2 + hidden_size
        self.geo_norm = nn.LayerNorm(hidden_size)
        self.geo_proj = nn.Sequential(
            nn.Linear(hidden_size, geo_hidden_dim),
            nn.GELU(),
            nn.Linear(geo_hidden_dim, self.geo_out_dim),
        )

        # Dynamic routing (H_pre, H_post)
        self.route_norm = nn.LayerNorm(hidden_size)
        self.route_proj = nn.Linear(hidden_size, 2 * n * n, bias=True)

        self._init_weights(init_gate_bias)

        self.register_buffer('I', torch.eye(n_streams))
        self._gate_reg_loss = None

    def _init_weights(self, init_gate_bias):
        with torch.no_grad():
            nn.init.normal_(self.geo_proj[0].weight, std=0.02)
            nn.init.zeros_(self.geo_proj[0].bias)
            nn.init.zeros_(self.geo_proj[2].weight)
            bias = self.geo_proj[2].bias
            bias[:-1].zero_()
            bias[-1].fill_(init_gate_bias)

            nn.init.zeros_(self.route_proj.weight)
            self.route_proj.bias.zero_()

    def forward(self, x_streams, layer_fn, rms_norm_eps):
        """
        Args:
            x_streams: (B, T, n, d) multi-stream input
            layer_fn: attention or FFN function
            rms_norm_eps: epsilon for RMS normalization
        Returns:
            x_out: (B, T, n, d)
        """
        B, T, n, d = x_streams.shape
        D = self.hidden_size

        x_flat = x_streams.reshape(B, T, D)

        # === Geometric operator ===
        x_pooled = x_flat.mean(dim=1)  # (B, D)
        x_gnorm = self.geo_norm(x_pooled)
        params = self.geo_proj(x_gnorm)  # (B, geo_out_dim)

        idx = 0
        u = params[:, idx:idx+n]; idx += n
        v = params[:, idx:idx+n]; idx += n
        beta = F.softplus(params[:, idx:idx+1]); idx += 1
        k_raw = params[:, idx:idx+D]; idx += D
        gate_logit = params[:, idx:idx+1]

        k = F.normalize(k_raw + 1e-8, dim=-1)
        gamma = torch.sigmoid(gate_logit)

        self._gate_reg_loss = self.gate_reg_weight * 4 * gamma * (1 - gamma)
        self._gate_reg_loss = self._gate_reg_loss.mean()

        gamma_bc = gamma.view(B, 1, 1, 1)

        # Cayley rotation on streams (exact, n×n)
        A = torch.einsum('bi,bj->bij', u, v) - torch.einsum('bi,bj->bij', v, u)
        M = (beta.view(B, 1, 1) / 2) * A
        I_batch = self.I.unsqueeze(0).expand(B, -1, -1)
        Q = torch.linalg.solve(I_batch + M, I_batch - M)
        x_rotated = torch.einsum('bnm,btmd->btnd', Q, x_streams)

        # Householder reflection on FULL dimension
        k_dot_x = (k.unsqueeze(1) * x_flat).sum(dim=-1, keepdim=True)  # (B, T, 1)
        x_reflected = x_flat - 2 * k_dot_x * k.unsqueeze(1)  # (B, T, D)
        x_reflected = x_reflected.reshape(B, T, n, d)

        x_geo = gamma_bc * x_rotated + (1 - gamma_bc) * x_reflected

        # === Dynamic routing around F ===
        x_rnorm = self.route_norm(x_geo.reshape(B, T, D))
        route = self.route_proj(x_rnorm).view(B, T, 2, n, n)
        H_pre = F.softmax(route[:, :, 0], dim=-1)
        H_post = F.softmax(route[:, :, 1], dim=-2)

        x_pre = torch.einsum('btij,btjd->btid', H_pre, x_geo)
        x_agg = x_pre.mean(dim=2)  # (B, T, d)

        y = layer_fn(x_agg)
        y_bc = y.unsqueeze(2).expand(-1, -1, n, -1)
        x_post = torch.einsum('btij,btjd->btid', H_post, y_bc)

        x_out = x_geo + x_post
        return x_out

    def get_gate_regularization_loss(self):
        if self._gate_reg_loss is None:
            return torch.tensor(0.0)
        return self._gate_reg_loss
