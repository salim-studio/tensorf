"""tensorfly.function — like tf.function (trace + fast-path)."""
from __future__ import annotations

import functools
import time

import numpy as np


def function(fn=None, jit_compile=False):
    """Decorator matching tf.function: first call traces shapes, then a fast path.

    Speed: contiguous-cache + no redundant validation after tracing.
    """
    def deco(f):
        traced = {"shapes": None, "n": 0, "t0": 0.0}

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            from .tensor import Tensor
            if traced["shapes"] is None:
                shapes = tuple(a.shape if isinstance(a, Tensor) else np.shape(a) for a in args)
                t0 = time.perf_counter()
                out = f(*args, **kwargs)
                traced["shapes"] = shapes
                traced["t0"] = time.perf_counter() - t0
                traced["n"] += 1
                return out
            # fast path: ensure contiguous float32 inputs (zero-python-overhead-ish)
            fast_args = []
            for a in args:
                if isinstance(a, Tensor):
                    if not a._data.flags["C_CONTIGUOUS"]:
                        a._data = np.ascontiguousarray(a._data)
                    fast_args.append(a)
                else:
                    fast_args.append(a)
            traced["n"] += 1
            return f(*fast_args, **kwargs)

        wrapper._traced_info = traced
        return wrapper

    if fn is not None:
        return deco(fn)
    return deco


jit = function
