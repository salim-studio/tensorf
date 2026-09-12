"""tensorfly.random — like tf.random."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor

_seed = None


def set_seed(s):
    global _seed
    _seed = s
    np.random.seed(s)


def _rng(seed):
    return np.random.default_rng(seed if seed is not None else _seed)


def normal(shape, mean=0.0, stddev=1.0, dtype=np.float32, seed=None) -> Tensor:
    return Tensor((_rng(seed).standard_normal(tuple(shape) if isinstance(shape, (list, tuple)) else shape) * stddev + mean).astype(dtype or np.float32))


def uniform(shape, minval=0, maxval=1, dtype=np.float32, seed=None) -> Tensor:
    return Tensor(_rng(seed).uniform(minval, maxval, tuple(shape) if isinstance(shape, (list, tuple)) else shape).astype(dtype or np.float32))


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32, seed=None) -> Tensor:
    r = _rng(seed)
    v = r.standard_normal(tuple(shape) if isinstance(shape, (list, tuple)) else shape)
    v = np.clip(v, -2, 2) * stddev + mean
    return Tensor(v.astype(dtype or np.float32))


def randint(shape, minval, maxval, dtype=np.int32, seed=None) -> Tensor:
    return Tensor(_rng(seed).integers(minval, maxval, tuple(shape) if isinstance(shape, (list, tuple)) else shape).astype(dtype))


def shuffle(t, seed=None) -> Tensor:
    from .tensor import convert_to_tensor
    a = (t._data if isinstance(t, Tensor) else np.asanyarray(t)).copy()
    _rng(seed).shuffle(a)
    return Tensor(a)


randn = normal
rand = uniform
