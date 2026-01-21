"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class GeodesicDelta(nn.Module):
    """
    The Geodesic Manifold-Delta Operator (E-Delta-MHC-Geo).
    Performs unitary rotation on the residual stream gated by thermodynamic entropy.
    """
    def __init__(self, d_model, n_streams=4, init_bias=-6.0):
        super().__init__()
        assert d_model % n_streams == 0, f"d_model {d_model} must be divisible by n_streams {n_streams}"

        self.d_model = d_model
        self.n_streams = n_streams
        self.d_stream = d_model // n_streams

        # 1. Learnable Generator Vectors (u, v) -> Defines rotation plane
        # Shape: (1, 1, n_streams, 1) broadcastable
        self.u = nn.Parameter(torch.randn(1, 1, n_streams, 1) * 0.01)
        self.v = nn.Parameter(torch.randn(1, 1, n_streams, 1) * 0.01)

        # 2. Thermodynamic Gating Parameters
        self.w_alpha = nn.Parameter(torch.zeros(1))
        self.b_init = nn.Parameter(torch.tensor(init_bias))

    def get_purity_proxy(self, x_streams):
        # x_streams: (B, S, n, d_stream)
        # Compute Gram Matrix G = X @ X.T
        G = torch.matmul(x_streams, x_streams.transpose(-1, -2))  # (B, S, n, n)

        # Frobenius Norm & Trace
        frob_sq = torch.sum(G**2, dim=(-1, -2), keepdim=True)
        tr = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        tr_sq = tr ** 2 + 1e-6

        # Purity Proxy Phi
        phi = 1.0 - (frob_sq / tr_sq)
        return phi

    def forward(self, x, return_beta=False):
        B, S, D = x.shape
        # View as N streams
        x_streams = x.view(B, S, self.n_streams, self.d_stream)

        # Thermodynamic Gate
        phi = self.get_purity_proxy(x_streams)
        beta = F.softplus(self.w_alpha * phi + self.b_init)

        # Construct Skew-Symmetric Generator A = uv^T - vu^T
        term1 = torch.matmul(self.u, self.v.transpose(-1, -2))
        term2 = torch.matmul(self.v, self.u.transpose(-1, -2))
        A = term1 - term2  # (1, 1, n, n)

        # Cayley Transform: Q = (I+M)^-1 (I-M) where M = beta*A
        M = beta * A
        I = torch.eye(self.n_streams, device=x.device).view(1, 1, self.n_streams, self.n_streams)

        numerator = I - M
        denominator = I + M

        # Solve linear system for stability
        Q = torch.linalg.solve(denominator, numerator)

        # Apply Rotation
        x_rotated = torch.matmul(Q, x_streams)
        x_out = x_rotated.view(B, S, D)

        if return_beta:
            # Squeeze beta from (B, S, 1, 1) to (B, S, 1) for broadcasting with (B, S, D)
            return x_out, beta.squeeze(-1)
        return x_out


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
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
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # --- NEW: Geodesic Components ---
        # n_streams=4 is a safe default. Ensure n_embd is divisible by 4.
        self.geo_attn = GeodesicDelta(config.n_embd, n_streams=4, init_bias=-6.0)
        self.geo_mlp = GeodesicDelta(config.n_embd, n_streams=4, init_bias=-6.0)

        # mHC Mixing Matrices (Identity Initialized)
        self.h_pre_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_attn = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_pre_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.h_post_mlp = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Force Identity Init for Mixing
        with torch.no_grad():
            self.h_pre_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_post_attn.weight.copy_(torch.eye(config.n_embd))
            self.h_pre_mlp.weight.copy_(torch.eye(config.n_embd))
            self.h_post_mlp.weight.copy_(torch.eye(config.n_embd))

    def forward(self, x):
        # --- ATTENTION BLOCK ---
        # 1. Geodesic Rotation & Thermodynamic Gate
        x_rotated, beta = self.geo_attn(x, return_beta=True)
        damper = 1.0 - torch.tanh(beta)

        # 2. Covariant Update (Function sees rotated state)
        normed = self.ln_1(x_rotated)
        mixed_input = self.h_pre_attn(normed)
        attn_out = self.attn(mixed_input)
        mixed_output = self.h_post_attn(attn_out)

        # 3. Apply Update
        x = x_rotated + (damper * mixed_output)

        # --- MLP BLOCK ---
        x_rotated, beta = self.geo_mlp(x, return_beta=True)
        damper = 1.0 - torch.tanh(beta)

        normed = self.ln_2(x_rotated)
        mixed_input = self.h_pre_mlp(normed)
        mlp_out = self.mlp(mixed_input)
        mixed_output = self.h_post_mlp(mlp_out)

        x = x_rotated + (damper * mixed_output)

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


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
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
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
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        NOTE: Loading pretrained GPT-2 weights is NOT supported for this model.

        This model contains additional Geodesic components (GeodesicDelta, mHC mixing
        matrices) that do not exist in the original GPT-2 architecture. The parameter
        count and state_dict keys differ significantly, making direct weight loading
        incompatible.

        To use this model, train from scratch using train.py.

        For pretrained GPT-2 weights, use the original model.py instead.
        """
        raise NotImplementedError(
            "from_pretrained() is not supported for the Geodesic model architecture. "
            "This model has additional components (GeodesicDelta, mHC mixing matrices) "
            "that do not exist in standard GPT-2. Please train from scratch or use "
            "the original model.py for pretrained weight loading."
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
