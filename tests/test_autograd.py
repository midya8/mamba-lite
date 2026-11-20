"""Every op in the engine gets a finite-difference check on a small shape."""
import numpy as np
import mambalite.autograd as ag


def fd(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def check(name, build, shape, tol=1e-5):
    rng = np.random.default_rng(abs(hash(name)) % 2**31)
    x = rng.normal(size=shape)
    node = ag.param(x)
    out = build(node)
    out.backward()
    ref = fd(lambda a: float(np.sum(build(ag.param(a)).v)), x)
    rel = np.abs(node.g - ref).max() / max(np.abs(ref).max(), 1e-12)
    assert rel < tol, f"{name}: rel err {rel:.2e}"


def test_pointwise():
    check("add", lambda a: ag.add(a, ag.const(0.5)), (4, 3))
    check("mul", lambda a: ag.mul(a, ag.const(2.0)), (4, 3))
    check("exp", lambda a: ag.exp(a), (5,))
    check("log", lambda a: ag.log(ag.exp(a)), (5,))
    check("sigmoid", lambda a: ag.sigmoid(a), (5,))
    check("softplus", lambda a: ag.softplus(a), (5,))
    check("silu", lambda a: ag.silu(a), (5,))


def test_matmul():
    rng = np.random.default_rng(2)
    w = ag.param(rng.normal(size=(3, 4)))
    check("matmul", lambda a: ag.matmul(ag.reshape(a, (5, 3)), w), (5, 3))


def test_reductions():
    check("sum_axis", lambda a: ag.sum(ag.mul(a, ag.const(1.0)), axis=2), (3, 4, 5))


def test_shape_ops():
    check("reshape", lambda a: ag.reshape(a, (6, 10)), (2, 3, 10))
    check("slice", lambda a: ag.slice(a, (slice(None), slice(1, 3), slice(None))),
          (4, 5, 6))
    check("stack", lambda a: ag.stack([a, ag.mul(a, ag.const(2.0))], axis=0), (3, 4))


def test_gather():
    rng = np.random.default_rng(3)
    w = ag.param(rng.normal(size=(7, 4)))
    idx = rng.integers(0, 7, size=(2, 9))
    out = ag.gather(w, idx.reshape(-1))
    out.backward()
    ref = fd(lambda a: float(np.sum(ag.gather(ag.param(a), idx.reshape(-1)).v)),
             w.v)
    rel = np.abs(w.g.ravel() - ref.ravel()).max() / max(np.abs(ref).max(), 1e-12)
    assert rel < 1e-5


def test_cross_entropy():
    rng = np.random.default_rng(4)
    logits = ag.param(rng.normal(size=(2, 3, 5)))
    targets = rng.integers(0, 5, size=(2, 3))
    mask = np.ones((2, 3))
    out = ag.cross_entropy(logits, targets, mask)
    out.backward()
    ref = fd(lambda a: float(ag.cross_entropy(ag.param(a), targets, mask).v),
             logits.v)
    rel = np.abs(logits.g - ref).max() / max(np.abs(ref).max(), 1e-12)
    assert rel < 1e-5


def test_add_broadcast_on_either_side():
    # regression: the gradient for the broadcast operand used to skip the
    # sum-to-shape step when the small tensor came first
    x = ag.param(np.arange(12.0).reshape(3, 4))
    w = ag.param(np.array([1.0, -1.0, 2.0, 0.5]))
    y = ag.add(ag.reshape(w, (1, 1, 4)), ag.reshape(x, (3, 1, 4)))
    y.backward()
    assert np.allclose(w.g, 3.0 * np.ones(4))
