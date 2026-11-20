"""Greedy generation stays consistent with the forward pass."""
import numpy as np
from mambalite.ssm import Model
from mambalite.data import sample_batch, SEP, FILL, EMIT, N_COPY, FILL_MAX


def test_generate_matches_forward():
    m = Model(rng=np.random.default_rng(3), d=8, d_inner=12, dt_rank=2,
              n_states=4, n_layers=1, conv_k=3, selective=True)
    rng = np.random.default_rng(11)
    ids, _, _ = sample_batch(rng, 1)
    prefix = ids[0, :8 + 1 + 6]  # data, separator, some fill
    outs = m.generate(prefix, 4, rng)
    assert len(outs) == 4
    full = np.concatenate([prefix, np.array(outs)])
    logits = m.forward(full[None, :]).v
    for k, tok in enumerate(outs):
        assert int(np.argmax(logits[0, len(prefix) + k])) == tok


def test_step_logits_matches_forward():
    m = Model(rng=np.random.default_rng(4), d=8, d_inner=12, dt_rank=2,
              n_states=4, n_layers=1, conv_k=3, selective=True)
    rng = np.random.default_rng(12)
    ids, _, _ = sample_batch(rng, 2)
    a = m.step_logits(ids).v
    b = m.forward(ids).v
    assert np.array_equal(a, b)
