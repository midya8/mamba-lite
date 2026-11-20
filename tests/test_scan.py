"""The graph-based model agrees with a plain-loop reference implementation."""
import numpy as np
import mambalite.autograd as ag
from mambalite.ssm import Model
from mambalite.data import sample_batch


def softplus_np(v):
    return np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)


def silu_np(v):
    return v / (1.0 + np.exp(-v))


def rmsnorm_np(x, w, eps=1e-5):
    xc = x - x.mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + eps)
    return xc * rstd * w


def conv_np(xb, w, b):
    B, T, C = xb.shape
    K = w.shape[0]
    xp = np.zeros((B, T + K - 1, C))
    xp[:, K - 1:] = xb
    return sum(xp[:, k:k + T] * w[k][None, None] for k in range(K)) + b[None, None]


def reference_logits(params, ids, selective=True):
    B, T = ids.shape
    d = params["embed"].v.shape[1]
    di = params["l0_wx"].v.shape[1]
    N = params["l0_A_log"].v.shape[1]
    n_layers = sum(1 for k in params if k.startswith("norm")) - 1
    x = params["embed"].v[ids.reshape(-1)].reshape(B, T, d)
    for i in range(n_layers):
        pre = f"l{i}_"
        xn = rmsnorm_np(x, params[f"norm{i}"].v)
        xb = xn @ params[pre + "wx"].v
        z = xn @ params[pre + "wz"].v
        xc = silu_np(conv_np(xb, params[pre + "conv_w"].v, params[pre + "conv_b"].v))
        if selective:
            dbl = xc @ params[pre + "x_proj"].v
            R = params[pre + "dt_proj"].v.shape[0]
            dt = softplus_np(dbl[..., :R] @ params[pre + "dt_proj"].v
                             + params[pre + "dt_bias"].v)
            Bm, Cm = dbl[..., R:R + N], dbl[..., R + N:R + 2 * N]
        else:
            dt = np.broadcast_to(softplus_np(params[pre + "dt_bias"].v),
                                 (B, T, di))
            Bm = np.broadcast_to(params[pre + "B0"].v, (B, T, N))
            Cm = np.broadcast_to(params[pre + "C0"].v, (B, T, N))
        A = -np.exp(params[pre + "A_log"].v)
        a = np.exp(dt[:, :, :, None] * A[None, None])     # (B,T,di,N)
        b = dt[:, :, :, None] * Bm[:, :, None, :] * xb[:, :, :, None]
        h = np.zeros((B, di, N))
        ys = np.zeros((B, T, di))
        for t in range(T):
            h = a[:, t] * h + b[:, t]                        # (B,di,N)
            ys[:, t] = (Cm[:, t][:, None, :] * h).sum(axis=2)
        y = ys + params[pre + "D"].v[None, None] * xb
        y = y * silu_np(z)
        x = y @ params[pre + "out_proj"].v + x
    x = rmsnorm_np(x, params[f"norm{n_layers}"].v)
    return x @ params["head"].v


def test_forward_matches_reference():
    rng = np.random.default_rng(7)
    for selective in (True, False):
        m = Model(rng=np.random.default_rng(1), d=8, d_inner=12, dt_rank=2,
                  n_states=4, n_layers=2, conv_k=3, selective=selective)
        ids, targets, mask = sample_batch(rng, 2)
        logits = m.forward(ids).v
        ref = reference_logits(m.params, ids, selective=selective)
        assert np.abs(logits - ref).max() < 1e-8, selective


def test_loss_matches_reference():
    rng = np.random.default_rng(9)
    m = Model(rng=np.random.default_rng(2), d=8, d_inner=12, dt_rank=2,
              n_states=4, n_layers=1, conv_k=3, selective=True)
    ids, targets, mask = sample_batch(rng, 2)
    loss = float(m.forward(ids, targets, mask)[0].v)
    logits = reference_logits(m.params, ids)
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    sel = mask > 0
    nll = -np.log(p[np.arange(2)[:, None], np.arange(logits.shape[1]),
                    targets][sel]).sum() / sel.sum()
    assert abs(loss - nll) < 1e-8
