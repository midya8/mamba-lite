"""Adam with weight decay off: enough for these sizes."""
import numpy as np


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p.v) for p in params]
        self.v = [np.zeros_like(p.v) for p in params]
        self.t = 0

    def step(self, max_norm=None):
        self.t += 1
        if max_norm is not None:
            gs = [p.g for p in self.params if p.g is not None]
            if gs:
                total = np.sqrt(sum(float(np.sum(g * g)) for g in gs))
                if total > max_norm:
                    scale = max_norm / total
                    for g in gs:
                        g *= scale
        for i, p in enumerate(self.params):
            g = p.g
            if g is None:
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.v -= self.lr * mh / (np.sqrt(vh) + self.eps)
            p.g = None
