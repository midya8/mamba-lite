"""A tiny reverse-mode autograd engine, written for this repo.

Only the ops the model needs. Each op records a backward closure; backward()
walks the graph once in reverse topological order. Float64 everywhere so
finite-difference checks agree with analytic gradients to many decimals.
"""
import numpy as np


def _unbroadcast(g, shape):
    # Sum gradient down to the target shape after a broadcast op.
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Node:
    __slots__ = ("v", "g", "parents", "requires_grad")

    def __init__(self, v, parents=(), requires_grad=True):
        self.v = v
        self.g = None
        self.parents = parents  # (node, fn) where fn maps d(self) -> d(node)
        self.requires_grad = requires_grad

    def backward(self):
        order = []
        seen = set()
        stack = [(self, False)]
        while stack:
            n, done = stack.pop()
            if done:
                order.append(n)
                continue
            if id(n) in seen:
                continue
            seen.add(id(n))
            stack.append((n, True))
            for p, _ in n.parents:
                if id(p) not in seen:
                    stack.append((p, False))
        self.g = np.ones_like(self.v)
        for n in reversed(order):
            if n.g is None:
                continue
            for p, fn in n.parents:
                if p.requires_grad:
                    gp = fn(n.g)
                    p.g = gp if p.g is None else p.g + gp


def const(a):
    return Node(np.asarray(a, dtype=np.float64), requires_grad=False)


def param(a, name=None):
    return Node(np.asarray(a, dtype=np.float64))


def _wrap(x):
    return x if isinstance(x, Node) else const(x)


def add(a, b):
    a, b = _wrap(a), _wrap(b)
    out = Node(a.v + b.v, [(a, lambda g: _unbroadcast(g, a.v.shape)),
                           (b, lambda g: _unbroadcast(g, b.v.shape))])
    return out


def mul(a, b):
    a, b = _wrap(a), _wrap(b)
    out = Node(a.v * b.v, [(a, lambda g: _unbroadcast(g * b.v, a.v.shape)),
                           (b, lambda g: _unbroadcast(g * a.v, b.v.shape))])
    return out


def matmul(a, b):
    """A @ B with leading batch dims broadcast: A (..., n, k), B (k, m)."""
    a, b = _wrap(a), _wrap(b)
    out = Node(a.v @ b.v, [
        (a, lambda g: g @ b.v.T),
        (b, lambda g: _grad_b(a.v, g))])
    return out


def _grad_b(a_v, g):
    if a_v.ndim == 2:
        return a_v.T @ g
    av = a_v.reshape(-1, a_v.shape[-2], a_v.shape[-1])
    gv = g.reshape(-1, g.shape[-2], g.shape[-1])
    return (av.transpose(0, 2, 1) @ gv).sum(axis=0)
    return out


def exp(a):
    a = _wrap(a)
    return Node(np.exp(a.v), [(a, lambda g: g * np.exp(a.v))])


def log(a):
    a = _wrap(a)
    return Node(np.log(a.v), [(a, lambda g: g / a.v)])


def sigmoid(a):
    a = _wrap(a)
    s = 1.0 / (1.0 + np.exp(-a.v))
    return Node(s, [(a, lambda g: g * s * (1.0 - s))])


def softplus(a):
    a = _wrap(a)
    v = a.v
    sp = np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)
    ds = 1.0 / (1.0 + np.exp(-v))
    return Node(sp, [(a, lambda g: g * ds)])


def silu(a):
    a = _wrap(a)
    s = 1.0 / (1.0 + np.exp(-a.v))
    v = a.v * s
    return Node(v, [(a, lambda g: g * s * (1.0 + a.v * (1.0 - s)))])


def sum(a, axis=None, keepdims=False):
    a = _wrap(a)
    if axis is None:
        out_v = np.array(a.v.sum())
        back = lambda g: np.full_like(a.v, float(g))
        return Node(out_v, [(a, back)])
    back = lambda g: np.broadcast_to(g, a.v.shape) if keepdims else np.broadcast_to(
        np.expand_dims(g, axis), a.v.shape)
    return Node(a.v.sum(axis=axis, keepdims=keepdims), [(a, back)])


def reshape(a, shape):
    a = _wrap(a)
    return Node(a.v.reshape(shape), [(a, lambda g: g.reshape(a.v.shape))])


def gather(a, idx):
    """Row gather: out[i] = a[idx[i]]."""
    a = _wrap(a)
    return Node(a.v[idx], [(a, lambda g: _gather_back(a, idx, g))])


def _gather_back(a, idx, g):
    full = np.zeros_like(a.v)
    np.add.at(full, idx, g)
    return full


def cross_entropy(logits, targets, mask=None):
    """Mean over masked positions of -log softmax(logits) at the target id.

    logits: (B, T, V) node. targets: (B, T) ints. mask: (B, T) float in {0, 1}.
    Returns a scalar node.
    """
    B, T, V = logits.v.shape
    x = logits.v.reshape(B * T, V)
    t = targets.reshape(B * T)
    m = np.ones(B * T) if mask is None else mask.reshape(B * T)
    mx = x.max(axis=1, keepdims=True)
    lse = mx + np.log(np.exp(x - mx).sum(axis=1, keepdims=True))
    lp = x[np.arange(len(t)), t] - lse[:, 0]
    n = float(m.sum()) or 1.0
    val = -float((m * lp).sum()) / n

    def back(g):
        sm = np.exp(x - lse)
        sm[np.arange(len(t)), t] -= 1.0
        return (sm * (m * float(g) / n)[:, None]).reshape(B, T, V)

    return Node(np.array(val), [(logits, back)])


def _logsumexp(x):
    mx = x.max(axis=1, keepdims=True)
    return (mx + np.log(np.exp(x - mx).sum(axis=1, keepdims=True))).ravel()


def _lse_backward(x, g, t, m):
    # d/dx of lse_i = softmax(x_i); weight each row by its mask and upstream grad
    mx = x.max(axis=1, keepdims=True)
    e = np.exp(x - mx)
    sm = e / e.sum(axis=1, keepdims=True)
    w = (m * g.ravel())[:, None]
    return (sm * w).reshape(x.shape)


def slice(a, sel):
    """Numpy-style slicing that keeps gradients flowing to the source node."""
    a = _wrap(a)
    out = Node(a.v[sel], [(a, lambda g: _slice_back(a, sel, g))])
    return out


def _slice_back(a, sel, g):
    full = np.zeros_like(a.v)
    full[sel] = g
    return full


def stack(nodes, axis=0):
    vals = [n.v for n in nodes]
    out_v = np.stack(vals, axis=axis)

    import functools
    parents = [(n, functools.partial(_take_i, i=i, axis=axis)) for i, n in enumerate(nodes)]
    return Node(out_v, parents)


def _take_i(g, i=0, axis=0):
    return np.take(g, i, axis=axis)


def conv1d_causal(x, w, b):
    """Causal depthwise conv over time. x: (B, T, C) node, w: (K, C) node,
    b: (C,) node. out[b, t, c] = sum_k w[k, c] * x[b, t + k - K + 1, c] + b[c],
    with x zero-padded on the left.
    """
    B, T, C = x.v.shape
    K, _ = w.v.shape
    xp = np.zeros((B, T + K - 1, C))
    xp[:, K - 1:, :] = x.v
    out_v = np.zeros((B, T, C))
    for k in range(K):
        out_v += xp[:, k:k + T, :] * w.v[k][None, None, :]
    out_v += b.v[None, None, :]

    def back(g):
        gx = np.zeros_like(xp)
        gw = np.zeros_like(w.v)
        for k in range(K):
            gx[:, k:k + T, :] += g * w.v[k][None, None, :]
            gw[k] += (g * xp[:, k:k + T, :]).sum(axis=(0, 1))
        return gx[:, K - 1:, :], gw, g.sum(axis=(0, 1))

    return Node(out_v, [(x, lambda g: back(g)[0]),
                        (w, lambda g: back(g)[1]),
                        (b, lambda g: back(g)[2])])
