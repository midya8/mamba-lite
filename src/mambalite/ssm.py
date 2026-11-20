"""Selective state-space model (Mamba-style) in pure NumPy.

Diagonal SSM with input-dependent discretization:

    a_t = exp(-softplus(dt_t) * exp(A_log))
    b_t = softplus(dt_t) * B_t * x_t
    h_t = a_t * h_{t-1} + b_t
    y_t = sum_n C_t[n] * h_t[n] + D * x_t
    out = y * silu(z), wrapped in pre-norm RMSNorm layers

The recurrence is unrolled into the autograd graph, so gradients come from
reverse mode, not hand-derived adjoints. Everything is float64.
"""
import numpy as np

from .autograd import (Node, add, const, conv1d_causal, cross_entropy, exp,
                       gather, log, matmul, mul, param, reshape, silu,
                       slice as slice_op, softplus, stack, sum)


def rmsnorm(x, w, eps=1e-5):
    """RMS norm over the last axis, then a learned scale, built from the
    existing primitives so gradients come for free."""
    d = x.v.shape[-1]
    B, T = x.v.shape[0], x.v.shape[1]
    s = sum(x, axis=-1)
    mean = reshape(mul(s, const(1.0 / d)), (B, T, 1))
    xc = add(x, mul(const(-1.0), mean))
    v2 = mul(sum(mul(xc, xc), axis=-1), const(1.0 / d))
    rstd = reshape(exp(mul(const(-0.5), log(add(v2, const(eps))))), (B, T, 1))
    return mul(mul(xc, rstd), reshape(w, (1, 1, d)))


class Model:
    def __init__(self, vocab=15, d=64, d_inner=128, dt_rank=8, n_states=16,
                 n_layers=2, conv_k=4, rng=None, selective=True):
        self.rng = rng or np.random.default_rng(0)
        self.selective = selective
        self.d, self.d_inner, self.dt_rank = d, d_inner, dt_rank
        self.N, self.n_layers, self.conv_k = n_states, n_layers, conv_k
        self.vocab = vocab
        P = self.params = {}

        self.param("embed", (vocab, d), 1.0)
        self.param("head", (d, vocab), 0.02)
        for i in range(n_layers + 1):
            self.param(f"norm{i}", (d,), 1.0)
        for i in range(n_layers):
            pre = f"l{i}_"
            self.param(pre + "wx", (d, d_inner), 1.0 / np.sqrt(d))
            self.param(pre + "wz", (d, d_inner), 1.0 / np.sqrt(d))
            self.param(pre + "conv_w", (conv_k, d_inner), 1.0)
            self.param(pre + "conv_b", (d_inner,), 0.0)
            self.param(pre + "x_proj", (d_inner, dt_rank + 2 * n_states),
                       1.0 / np.sqrt(d_inner))
            self.param(pre + "dt_proj", (dt_rank, d_inner), 0.05)
            self.param(pre + "dt_bias", (d_inner,), 0.0)
            self.param(pre + "A_log", (d_inner, n_states), 0.0)
            self.param(pre + "D", (d_inner,), 0.0)
            # used only by the LTI ablation: fixed write/read vectors
            self.param(pre + "B0", (n_states,), 1.0 / np.sqrt(n_states))
            self.param(pre + "C0", (n_states,), 1.0 / np.sqrt(n_states))
            self.param(pre + "out_proj", (d_inner, d), 1.0 / np.sqrt(d_inner))
        # dt_bias and A_log get the usual Mamba init below, after creation.

    def param(self, name, shape, scale):
        v = self.rng.normal(0.0, scale, size=shape)
        if name.endswith("dt_bias"):
            # inverse softplus of dt in [0.001, 0.1], per Mamba's init
            u = self.rng.uniform(0.001, 0.1, size=shape)
            v = np.log(np.exp(u) - 1.0)
        if name.endswith("A_log"):
            n = np.arange(1, shape[1] + 1, dtype=np.float64)
            v = np.tile(np.log(n)[None, :], (shape[0], 1))
        self.params[name] = param(v)
        return self.params[name]

    def weights(self):
        return list(self.params.values())

    def forward(self, ids, targets=None, mask=None):
        """ids: (B, T) ints. Returns (loss, logits_node) or logits_node."""
        B, T = ids.shape
        d, di, N, R = self.d, self.d_inner, self.N, self.dt_rank
        x = gather(self.params["embed"], ids.reshape(-1))
        x = reshape(x, (B, T, d))
        for i in range(self.n_layers):
            pre = f"l{i}_"
            xn = rmsnorm(x, self.params[f"norm{i}"])
            xb = matmul(xn, self.params[pre + "wx"])                     # (B,T,di)
            z = matmul(xn, self.params[pre + "wz"])
            c = conv1d_causal(xb, self.params[pre + "conv_w"],
                              self.params[pre + "conv_b"])
            xc = silu(c)
            if self.selective:
                dbl = matmul(xc, self.params[pre + "x_proj"])            # (B,T,R+2N)
                dt_low = slice_op(dbl, (slice(None), slice(None), slice(0, R)))
                Bm = slice_op(dbl, (slice(None), slice(None), slice(R, R + N)))
                Cm = slice_op(dbl, (slice(None), slice(None), slice(R + N, R + 2 * N)))
                dt_p = matmul(dt_low, self.params[pre + "dt_proj"])      # (B,T,di)
                dt = softplus(add(dt_p, self.params[pre + "dt_bias"]))
            else:
                # LTI ablation: fixed rates and fixed write/read vectors, so the
                # only input-dependence left is x itself. The broadcasts below
                # materialize (B,T,*) so the reshape into 4-D works.
                zdi = Node(np.zeros((B, T, di)))
                zn = Node(np.zeros((B, T, N)))
                dt = add(softplus(reshape(self.params[pre + "dt_bias"], (1, 1, di))), zdi)
                Bm = add(reshape(self.params[pre + "B0"], (1, 1, N)), zn)
                Cm = add(reshape(self.params[pre + "C0"], (1, 1, N)), zn)
            A = mul(const(-1.0), exp(self.params[pre + "A_log"]))        # (di,N)
            dt4 = reshape(dt, (B, T, di, 1))
            a_t = exp(mul(dt4, reshape(A, (1, 1, di, N))))               # (B,T,di,N)
            x4 = reshape(xb, (B, T, di, 1))
            b4 = mul(mul(dt4, reshape(Bm, (B, T, 1, N))), x4)            # (B,T,di,N)
            hs = []
            h = Node(np.zeros((B, di, N)), requires_grad=False)
            for t in range(T):
                h = add(mul(slice_op(a_t, (slice(None), t, slice(None), slice(None))),
                            h),
                        slice_op(b4, (slice(None), t, slice(None), slice(None))))
                hs.append(h)
            hh = stack(hs, axis=1)                                       # (B,T,di,N)
            Cm3 = reshape(Cm, (B, T, 1, N))
            y = sum(mul(Cm3, hh), axis=3)                                # (B,T,di)
            y = add(y, mul(reshape(self.params[pre + "D"], (1, 1, di)), xb))
            y = mul(y, silu(z))
            x = add(matmul(y, self.params[pre + "out_proj"]), x)         # residual
        x = rmsnorm(x, self.params[f"norm{self.n_layers}"])
        logits = matmul(x, self.params["head"])                          # (B,T,V)
        if targets is None:
            return logits
        return cross_entropy(logits, targets, mask), logits

    def step_logits(self, ids):
        """Logits for autoregressive generation from a full prefix (B, t)."""
        return self.forward(ids)

    def generate(self, ids_prefix, n_new, rng):
        """Naive autoregressive generation: recompute the prefix each step.

        Kept simple on purpose; test_cache.py proves the math, not the speed.
        """
        seq = np.array(ids_prefix, dtype=np.int64)
        outs = []
        for _ in range(n_new):
            logits = self.forward(seq[None, :])
            nxt = int(np.argmax(logits.v[0, -1]))
            outs.append(nxt)
            seq = np.concatenate([seq, [nxt]])
        return outs
