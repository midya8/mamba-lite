"""Train the selective SSM on the copy task, or the same model with an
input-independent decay as the ablation. Writes a JSON result file."""
import argparse
import json
import time

import numpy as np

from .data import sample_batch
from .optim import Adam
from .ssm import Model


def evaluate(m, rng, n=256, batch=64):
    hits = 0
    total = 0
    exact = 0
    for _ in range(n // batch):
        ids, targets, mask = sample_batch(rng, batch)
        logits = m.forward(ids)
        pred = np.argmax(logits.v, axis=-1)
        good = (pred == targets) & (mask > 0)
        hits += good.sum()
        total += (mask > 0).sum()
        exact += (good.sum(axis=1) == (mask > 0).sum(axis=1)).sum()
    return float(hits / total), float(exact / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selective", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="assets/results.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m = Model(rng=rng, selective=bool(args.selective))
    opt = Adam(m.weights(), lr=args.lr)
    curve = []
    t0 = time.time()
    for step in range(args.steps):
        ids, targets, mask = sample_batch(rng, 64)
        loss, _ = m.forward(ids, targets, mask)
        loss.backward()
        opt.step(max_norm=1.0)
        if step % 100 == 0 or step == args.steps - 1:
            curve.append({"step": step, "loss": float(loss.v)})
        if step % 250 == 0 or step == args.steps - 1:
            acc, ex = evaluate(m, rng, n=128, batch=32)
            curve[-1]["token_acc"] = acc
            curve[-1]["seq_acc"] = ex
            print(f"{'sel' if args.selective else 'fix'} step {step:5d} "
                  f"loss {float(loss.v):8.4f} tok {acc:6.3f} seq {ex:6.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    acc, ex = evaluate(m, rng, n=256)
    rng2 = np.random.default_rng(args.seed + 77)
    np.save("assets/policy_test.npy" if False else "assets/w.npy",
            np.concatenate([p.v.ravel() for _, p in sorted(m.params.items())]))
    with open(args.out, "w") as f:
        json.dump({"selective": bool(args.selective), "steps": args.steps,
                   "seed": args.seed, "token_acc": acc, "seq_acc": ex,
                   "curve": curve}, f, indent=2)
    print("FINAL", "selective" if args.selective else "fixed",
          f"token {acc:.3f} seq {ex:.3f}")


if __name__ == "__main__":
    main()
