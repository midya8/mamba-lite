"""Selective copy: echo the tokens that came before the separator.

Sequence layout (token ids): B data tokens in 0..11, then SEP (id 12), then a
variable number of FILL tokens (id 13), then EMIT tokens (id 14), one per data
token. The loss counts only the EMIT block: the input there carries no answer,
so the model has to hold the first block in memory across a gap whose length
changes per sample. Predicting the token sitting right under you would be an
identity map, which is why the answer positions get EMIT instead.
"""
import numpy as np

SEP = 12
FILL = 13
EMIT = 14
VOCAB = 15
N_COPY = 8
FILL_MIN, FILL_MAX = 4, 12


def sample_batch(rng, batch_size, n_copy=N_COPY):
    fmax = FILL_MAX
    T = n_copy + 1 + fmax + n_copy
    ids = np.zeros((batch_size, T), dtype=np.int64)
    targets = np.zeros((batch_size, T), dtype=np.int64)
    mask = np.zeros((batch_size, T), dtype=np.float64)
    for b in range(batch_size):
        toks = rng.integers(0, 12, size=n_copy)
        f = int(rng.integers(FILL_MIN, fmax + 1))
        seq = np.concatenate([toks, [SEP], np.full(f, FILL),
                              np.full(fmax - f, FILL), np.full(n_copy, EMIT)])
        ids[b] = seq
        targets[b, T - n_copy:] = toks
        mask[b, T - n_copy:] = 1.0
    return ids, targets, mask
