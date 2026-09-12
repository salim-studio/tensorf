"""tensorfly.optimizers — like tf.keras.optimizers (with fast fused versions)."""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.iterations = 0

    def apply_gradients(self, grads_and_vars):
        raise NotImplementedError

    def _lr(self):
        lr = self.learning_rate
        return lr(self.iterations) if callable(lr) else float(lr)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self._vel = {}

    def apply_gradients(self, grads_and_vars):
        lr = self._lr()
        for g, v in grads_and_vars:
            if g is None:
                continue
            gd = g._data if hasattr(g, "_data") else np.asanyarray(g)
            if self.momentum > 0:
                key = id(v)
                vel = self._vel.get(key, np.zeros_like(v._data))
                vel = self.momentum * vel + gd
                self._vel[key] = vel
                step = vel * lr if not self.nesterov else (self.momentum * vel + gd) * lr
            else:
                step = gd * lr
            v._data -= step.astype(v._data.dtype, copy=False)
        self.iterations += 1


class Adam(Optimizer):
    """Fused Adam: a single update per variable with no extra temporaries."""

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._m, self._v = {}, {}

    def apply_gradients(self, grads_and_vars):
        lr = self._lr()
        self.iterations += 1
        t = self.iterations
        b1, b2 = self.beta_1, self.beta_2
        # fused bias correction scalar
        lr_t = lr * np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        for g, v in grads_and_vars:
            if g is None:
                continue
            gd = (g._data if hasattr(g, "_data") else np.asanyarray(g)).astype(np.float32)
            key = id(v)
            m = self._m.get(key)
            if m is None:
                m = np.zeros_like(v._data, dtype=np.float32)
                vv = np.zeros_like(v._data, dtype=np.float32)
                self._m[key], self._v[key] = m, vv
            vv = self._v[key]
            # fused in-place: m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
            np.multiply(m, b1, out=m)
            m += (1 - b1) * gd
            np.multiply(vv, b2, out=vv)
            vv += (1 - b2) * gd * gd
            v._data -= (lr_t * m / (np.sqrt(vv) + self.epsilon)).astype(v._data.dtype, copy=False)


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self._sq = {}

    def apply_gradients(self, grads_and_vars):
        lr = self._lr()
        for g, v in grads_and_vars:
            if g is None:
                continue
            gd = (g._data if hasattr(g, "_data") else np.asanyarray(g)).astype(np.float32)
            key = id(v)
            sq = self._sq.get(key)
            if sq is None:
                sq = np.zeros_like(v._data, dtype=np.float32)
                self._sq[key] = sq
            sq[:] = self.rho * sq + (1 - self.rho) * gd * gd
            v._data -= (lr * gd / (np.sqrt(sq) + self.epsilon)).astype(v._data.dtype, copy=False)
        self.iterations += 1


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self._acc = {}

    def apply_gradients(self, grads_and_vars):
        lr = self._lr()
        for g, v in grads_and_vars:
            if g is None:
                continue
            gd = (g._data if hasattr(g, "_data") else np.asanyarray(g)).astype(np.float32)
            key = id(v)
            acc = self._acc.get(key)
            if acc is None:
                acc = np.zeros_like(v._data, dtype=np.float32)
                self._acc[key] = acc
            acc += gd * gd
            v._data -= (lr * gd / (np.sqrt(acc) + self.epsilon)).astype(v._data.dtype, copy=False)
        self.iterations += 1


class AdamW(Adam):
    """Adam + decoupled weight decay (like TF-Addons/Optax)."""

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, weight_decay=0.01):
        super().__init__(learning_rate, beta_1, beta_2, epsilon)
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars):
        for g, v in grads_and_vars:
            if g is None:
                continue
            v._data -= float(self._lr()) * self.weight_decay * v._data
        super().apply_gradients(grads_and_vars)


class Adadelta(Optimizer):
    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho, self.epsilon = rho, epsilon
        self._sq, self._delta = {}, {}

    def apply_gradients(self, grads_and_vars):
        lr = self._lr()
        for g, v in grads_and_vars:
            if g is None:
                continue
            gd = (g._data if hasattr(g, "_data") else np.asanyarray(g)).astype(np.float32)
            key = id(v)
            sq = self._sq.get(key)
            if sq is None:
                sq = np.zeros_like(v._data, dtype=np.float32)
                dl = np.zeros_like(v._data, dtype=np.float32)
                self._sq[key], self._delta[key] = sq, dl
            dl = self._delta[key]
            sq[:] = self.rho * sq + (1 - self.rho) * gd * gd
            step = np.sqrt(dl + self.epsilon) / np.sqrt(sq + self.epsilon) * gd * lr
            dl[:] = self.rho * dl + (1 - self.rho) * step * step
            v._data -= step.astype(v._data.dtype, copy=False)
        self.iterations += 1


class Lion(Optimizer):
    """Evolved Sign Momentum (simplified) — effective for Transformers."""

    def __init__(self, learning_rate=1e-4, beta_1=0.9, beta_2=0.99, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta_1, self.beta_2 = beta_1, beta_2
        self.weight_decay = weight_decay
        self._m = {}

    def apply_gradients(self, grads_and_vars):
        lr = self._lr()
        for g, v in grads_and_vars:
            if g is None:
                continue
            gd = (g._data if hasattr(g, "_data") else np.asanyarray(g)).astype(np.float32)
            key = id(v)
            m = self._m.get(key)
            if m is None:
                m = np.zeros_like(v._data, dtype=np.float32)
                self._m[key] = m
            upd = np.sign(self.beta_1 * m + (1 - self.beta_1) * gd)
            m[:] = self.beta_2 * m + (1 - self.beta_2) * gd
            v._data -= (lr * (upd + self.weight_decay * v._data)).astype(v._data.dtype, copy=False)
        self.iterations += 1
