"""tensorf.schedules — learning-rate schedules (work with any optimizer via callable lr)."""
from __future__ import annotations

import numpy as np


class Schedule:
    def __call__(self, step: int) -> float:
        raise NotImplementedError


class Constant(Schedule):
    def __init__(self, lr):
        self.lr = lr
    def __call__(self, step):
        return float(self.lr)


class ExponentialDecay(Schedule):
    def __init__(self, initial_lr, decay_steps, decay_rate, staircase=False):
        self.l0, self.ds, self.dr, self.st = initial_lr, decay_steps, decay_rate, staircase
    def __call__(self, step):
        p = (step // self.ds) if self.st else (step / self.ds)
        return float(self.l0 * self.dr ** p)


class StepDecay(Schedule):
    def __init__(self, initial_lr, drop_every, factor=0.5):
        self.l0, self.de, self.f = initial_lr, drop_every, factor
    def __call__(self, step):
        return float(self.l0 * self.f ** (step // self.de))


class CosineDecay(Schedule):
    def __init__(self, initial_lr, decay_steps, alpha=0.0):
        self.l0, self.ds, self.a = initial_lr, decay_steps, alpha
    def __call__(self, step):
        s = min(step, self.ds)
        cos = 0.5 * (1 + np.cos(np.pi * s / self.ds))
        return float(self.a + (self.l0 - self.a) * cos)


class PolynomialDecay(Schedule):
    def __init__(self, initial_lr, decay_steps, end_lr=1e-5, power=1.0):
        self.l0, self.ds, self.le, self.p = initial_lr, decay_steps, end_lr, power
    def __call__(self, step):
        s = min(step, self.ds)
        return float((self.l0 - self.le) * (1 - s / self.ds) ** self.p + self.le)


class WarmupCosine(Schedule):
    def __init__(self, peak_lr, warmup_steps, decay_steps):
        self.pk, self.wu, self.ds = peak_lr, warmup_steps, decay_steps
        self._cos = CosineDecay(peak_lr, decay_steps)
    def __call__(self, step):
        if step < self.wu:
            return float(self.pk * (step + 1) / self.wu)
        return self._cos(step - self.wu)


__all__ = ["Schedule", "Constant", "ExponentialDecay", "StepDecay",
           "CosineDecay", "PolynomialDecay", "WarmupCosine"]
