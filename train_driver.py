"""Chunked training driver with checkpointing, for the selective-copy task."""
import argparse, json, time
import numpy as np
from mambalite.ssm import Model
from mambalite.data import sample_batch
from mambalite.optim import Adam

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=900)
ap.add_argument("--selective", type=int, default=1)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target", type=int, default=2000)
ap.add_argument("--tag", default="selective")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
m = Model(rng=rng, d=32, d_inner=64, dt_rank=4, n_states=8, n_layers=2,
          conv_k=4, selective=bool(args.selective))
opt = Adam(m.weights(), lr=args.lr)

ckpt = f"assets/ckpt_{args.tag}.npz"
step0, history = 0, []
try:
    z = np.load(ckpt, allow_pickle=True)
    for i, (k, p) in enumerate(m.params.items()):
        p.v[...] = z[k]
        p.g = None
        opt.m[i] = z[k + "|m"].copy()
        opt.v[i] = z[k + "|v"].copy()
    opt.t = int(z["t"]); step0 = int(z["step"])
    history = json.loads(str(z["history"]))
    print(f"resumed at step {step0}")
except FileNotFoundError:
    print("fresh run")

def evaluate(n=128):
    ev = np.random.default_rng(777)
    hits = total = exact = 0
    for _ in range(n // 32):
        ids, targets, mask = sample_batch(ev, 32)
        pred = np.argmax(m.forward(ids).v, axis=-1)
        good = (pred == targets) & (mask > 0)
        hits += good.sum(); total += (mask > 0).sum()
        exact += (good.sum(axis=1) == (mask > 0).sum(axis=1)).sum()
    return hits / total, exact / n

t0 = time.time()
for i in range(args.steps):
    step = step0 + i + 1
    ids, targets, mask = sample_batch(rng, 64)
    loss, _ = m.forward(ids, targets, mask)
    loss.backward()
    opt.step(max_norm=1.0)
    if step % 100 == 0:
        payload = {k: p.v for k, p in m.params.items()}
        for (k, p), mi, vi in zip(m.params.items(), opt.m, opt.v):
            payload[k + "|m"] = mi; payload[k + "|v"] = vi
        payload["t"] = np.array(opt.t); payload["step"] = np.array(step)
        payload["history"] = np.array(json.dumps(history))
        np.savez(ckpt, **payload)
        tok, seq = evaluate()
        history.append([step, round(float(loss.v), 4), round(float(tok), 4), round(float(seq), 4)])
        print(f"step {step} loss {float(loss.v):.4f} tok {tok:.3f} seq {seq:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if step >= args.target:
        break

payload = {k: p.v for k, p in m.params.items()}
for (k, p), mi, vi in zip(m.params.items(), opt.m, opt.v):
    payload[k + "|m"] = mi; payload[k + "|v"] = vi
payload["t"] = np.array(opt.t); payload["step"] = np.array(step)
payload["history"] = np.array(json.dumps(history))
np.savez(ckpt, **payload)
json.dump({"tag": args.tag, "selective": bool(args.selective), "step": step,
           "history": history}, open(f"assets/hist_{args.tag}.json", "w"))
if step >= args.target:
    np.save(f"assets/w_{args.tag}.npy", np.array([1]))  # marker
    z = np.load(ckpt)
    w = {k: z[k] for k in m.params}
    np.savez(f"assets/w_{args.tag}.npz", **w)
print(f"saved at step {step}")
