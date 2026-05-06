"""
Modified TRM Block with pluggable mixer modules.

Extends the original TRM Block to support multi-stream residual connections
via JPmHC or E∆ mixer modules. The mixer wraps each sub-block (attention
and FFN), replacing the standard identity residual with orthogonal mixing.

Three modes:
- 'none': Standard TRM (no mixer, original architecture)
- 'jpmhc': JPmHC mixer (iterative finite Cayley component)
- 'edelta': E∆ mixer (exact Cayley + Householder gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import Attention, SwiGLU, rms_norm, CosSin
from src.arc_agi.mixers import JPmHCMixer, EdeltaMixer


class MixedTRMBlock(nn.Module):
    """
    TRM Block with optional multi-stream mixer.

    Standard TRM: hidden = rms_norm(hidden + attn(hidden))
    With mixer:   hidden_streams = mixer(hidden_streams, attn_at_d_stream)
    """

    def __init__(self, config, mixer_type='none', mixer_kwargs=None):
        super().__init__()
        self.config = config
        self.mixer_type = mixer_type
        self.n_streams = mixer_kwargs.get('n_streams', 4) if mixer_kwargs else 4
        self.norm_eps = config.rms_norm_eps

        if config.mlp_t:
            raise NotImplementedError("MLP-T variant not supported with mixers")

        # Attention at d_stream width (per-stream compute)
        d_stream = config.hidden_size // self.n_streams if mixer_type != 'none' else config.hidden_size

        if mixer_type != 'none':
            # IMPORTANT: head_dim must match the parent TRM's RoPE dimension
            # (hidden_size // num_heads) so cos_sin can be shared.
            # E.g., hidden=512, parent_heads=8 → parent_head_dim=64
            # d_stream=128 → n_heads = d_stream // parent_head_dim = 2
            parent_head_dim = config.hidden_size // config.num_heads
            n_heads = max(1, d_stream // parent_head_dim)
            head_dim = parent_head_dim
        else:
            n_heads = config.num_heads
            head_dim = d_stream // n_heads

        self.self_attn = Attention(
            hidden_size=d_stream,
            head_dim=head_dim,
            num_heads=n_heads,
            num_key_value_heads=n_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=d_stream,
            expansion=config.expansion,
        )

        # Mixer modules (one per sub-block)
        if mixer_type == 'jpmhc':
            mkw = mixer_kwargs or {}
            self.attn_mixer = JPmHCMixer(config.hidden_size, self.n_streams, **{
                k: v for k, v in mkw.items() if k in ['cayley_alpha', 'cayley_iters']
            })
            self.ffn_mixer = JPmHCMixer(config.hidden_size, self.n_streams, **{
                k: v for k, v in mkw.items() if k in ['cayley_alpha', 'cayley_iters']
            })
        elif mixer_type == 'edelta':
            mkw = mixer_kwargs or {}
            self.attn_mixer = EdeltaMixer(config.hidden_size, self.n_streams, **{
                k: v for k, v in mkw.items()
                if k in ['geo_hidden_dim', 'gate_reg_weight', 'init_gate_bias']
            })
            self.ffn_mixer = EdeltaMixer(config.hidden_size, self.n_streams, **{
                k: v for k, v in mkw.items()
                if k in ['geo_hidden_dim', 'gate_reg_weight', 'init_gate_bias']
            })

    def _wrap_attn_for_mixer(self, cos_sin):
        """Create a callable that applies attention to d_stream input."""
        def attn_fn(x):
            # x: (B, T, d_stream) — already at per-stream width
            return self.self_attn(cos_sin=cos_sin, hidden_states=x)
        return attn_fn

    def _wrap_mlp_for_mixer(self):
        """Create a callable that applies FFN to d_stream input."""
        def mlp_fn(x):
            return self.mlp(x)
        return mlp_fn

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden_states.shape

        if self.mixer_type == 'none':
            # Standard TRM block (post-norm)
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps)
            out = self.mlp(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            return hidden_states

        # Multi-stream mode
        n, d = self.n_streams, D // self.n_streams
        x_streams = hidden_states.reshape(B, T, n, d)

        # Attention sub-block with mixer
        x_streams = self.attn_mixer(
            x_streams, self._wrap_attn_for_mixer(cos_sin), self.norm_eps)
        x_streams = rms_norm(x_streams.reshape(B, T, D),
                            variance_epsilon=self.norm_eps).reshape(B, T, n, d)

        # FFN sub-block with mixer
        x_streams = self.ffn_mixer(
            x_streams, self._wrap_mlp_for_mixer(), self.norm_eps)
        hidden_states = rms_norm(x_streams.reshape(B, T, D),
                                variance_epsilon=self.norm_eps)

        return hidden_states

    def get_gate_regularization_loss(self):
        """Sum gate regularization from both E∆ mixers."""
        loss = torch.tensor(0.0)
        if self.mixer_type == 'edelta':
            loss = (self.attn_mixer.get_gate_regularization_loss() +
                    self.ffn_mixer.get_gate_regularization_loss())
        return loss
