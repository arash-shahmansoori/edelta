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
from models.layers import rms_norm, CastedLinear


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

        self.fused_proj = CastedLinear(hidden_size, 3 * n_streams * n_streams, bias=True)

        with torch.no_grad():
            self.fused_proj.weight.zero_()
            self.fused_proj.bias.zero_()

        self.register_buffer('I', torch.eye(n_streams))

    def iterative_cayley(self, H_tilde):
        W = H_tilde - H_tilde.transpose(-1, -2)
        I = self.I.to(W.dtype)
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
        fused = self.fused_proj(rms_norm(x_flat, 1e-5)).view(B, T, 3, n, n)

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
        self.geo_proj = nn.Sequential(
            CastedLinear(hidden_size, geo_hidden_dim, bias=True),
            nn.GELU(),
            CastedLinear(geo_hidden_dim, self.geo_out_dim, bias=True),
        )

        # Dynamic routing (H_pre, H_post)
        self.route_proj = CastedLinear(hidden_size, 2 * n * n, bias=True)

        self._init_weights(init_gate_bias)

        self.register_buffer('I', torch.eye(n_streams))
        self._gate_reg_loss = None

    def _init_weights(self, init_gate_bias):
        with torch.no_grad():
            self.geo_proj[0].weight.normal_(std=0.02)
            self.geo_proj[0].bias.zero_()
            self.geo_proj[2].weight.zero_()
            self.geo_proj[2].bias[:-1].zero_()
            self.geo_proj[2].bias[-1].fill_(init_gate_bias)

            self.route_proj.weight.zero_()
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

        # === Per-token geometric operator ===
        params = self.geo_proj(rms_norm(x_flat, 1e-5))  # (B, T, geo_out_dim)

        idx = 0
        u = params[:, :, idx:idx+n]; idx += n          # (B, T, n)
        v = params[:, :, idx:idx+n]; idx += n          # (B, T, n)
        beta = F.softplus(params[:, :, idx:idx+1]); idx += 1  # (B, T, 1)
        k_raw = params[:, :, idx:idx+D]; idx += D      # (B, T, D)
        gate_logit = params[:, :, idx:idx+1]            # (B, T, 1)

        k = F.normalize(k_raw + 1e-8, dim=-1)     # (B, T, D)
        gamma = torch.sigmoid(gate_logit)           # (B, T, 1)

        self._gate_reg_loss = self.gate_reg_weight * 4 * gamma * (1 - gamma)
        self._gate_reg_loss = self._gate_reg_loss.mean()
        self._last_gamma = gamma.detach().mean()

        gamma_bc = gamma.unsqueeze(-1)              # (B, T, 1, 1)

        # Per-token Cayley rotation on streams (exact, n×n per token)
        # Cast to float32 for linalg.solve (not supported in bfloat16)
        orig_dtype = u.dtype
        A = torch.einsum('bti,btj->btij', u.float(), v.float()) - torch.einsum('bti,btj->btij', v.float(), u.float())
        M = (beta.float().unsqueeze(-1) / 2) * A
        I_batch = self.I.expand(B, T, -1, -1)
        BT = B * T
        Q = torch.linalg.solve(
            (I_batch + M).reshape(BT, n, n),
            (I_batch - M).reshape(BT, n, n)
        ).reshape(B, T, n, n).to(orig_dtype)
        x_rotated = torch.einsum('btij,btjd->btid', Q, x_streams)

        # Per-token Householder reflection on FULL dimension
        k_dot_x = (k * x_flat).sum(dim=-1, keepdim=True)  # (B, T, 1)
        x_reflected = (x_flat - 2 * k_dot_x * k).reshape(B, T, n, d)

        x_geo = gamma_bc * x_rotated + (1 - gamma_bc) * x_reflected

        # === Dynamic routing around F ===
        route = self.route_proj(rms_norm(x_geo.reshape(B, T, D), 1e-5)).view(B, T, 2, n, n)
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
