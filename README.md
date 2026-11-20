# mamba-lite

A minimal Mamba-style selective SSM, written from scratch in NumPy with a
hand-written reverse-mode autograd engine. No PyTorch, no JAX, nothing but
numpy and math.

The point of the repo is the ablation: the same model trained two ways, one
with a selective (input-dependent) recurrence and one with the selectivity
removed (fixed rates, an LTI system). Selectivity is what makes a modern SSM
work and here you can watch that claim come true on a task that specifically
rewards it.

## The task

Selective copy. A sequence holds 8 random tokens in a known range, a
separator, then a stretch of filler tokens. The model has to echo the first
block back, but only after it sees an EMIT token that comes at the end of the
filler. Before EMIT it must output nothing, and the filler never tells you
what to copy, so the answer is carried purely in the state, up to 12 tokens
back. An echo-only model (or a seq2seq shortcut) can't solve this; a model
that forgets can't either.

## What's inside

- `src/mambalite/autograd.py` reverse-mode autograd: add, mul, matmul, exp,
  log, sigmoid, softplus, silu, sum, reshape, gather, slice, stack, a fused
  cross-entropy, and a causal depthwise conv. Every op has a finite-difference
  gradient check in the tests.
- `src/mambalite/ssm.py` the model. Per block: RMSNorm pre-norm, input
  projection, causal conv, SiLU gate, the selective scan
  `h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t` unrolled over time
  (float64, plain numpy), then an output projection and a residual.
- `src/mambalite/data.py` the selective copy task generator.
- `src/mambalite/optim.py` Adam with gradient clipping.
- `train.py` short-run trainer. `train_driver.py` chunks a longer run with
  checkpointing so it can resume after a timeout.
- `tests/` gradient checks, a plain-loop reference implementation of the full
  forward pass compared against the graph version, and generation consistency.

## Results

Both models trained identically for 1800 steps (d=64, 2 blocks), then
evaluated on 256 held-out samples:

| model         | token acc | full-sequence acc |
|---------------|-----------|-------------------|
| selective     | 0.996     | 0.969             |
| fixed rate    | 0.427     | 0.000             |

The fixed-rate variant learns to output the right distribution of tokens but
never completes a single sequence: without input-dependent gating it cannot
decide when to emit, or hold the answer through up to 12 filler tokens.

`assets/learning_curve.png` shows both runs. Training takes about 15 minutes
on a laptop CPU in float64.

## Usage

```
pip install numpy pytest
PYTHONPATH=src python3 -m pytest tests/ -q
PYTHONPATH=src python3 train_driver.py --selective --steps 1800 --tag sel
```

Checkpoints go in `assets/` and resume automatically with the same tag.
