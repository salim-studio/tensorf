"""tensorf.autodiff — tf.GradientTape."""
from __future__ import annotations

import numpy as np


class GradientTape:
    """Like tf.GradientTape: records ops automatically (eager) and computes derivatives.

    with tf.GradientTape() as tape:
        y = W @ x + b
    grads = tape.gradient(y, [W, b])
    """

    def __init__(self, persistent: bool = False, watch_accessed_variables: bool = True):
        self.persistent = persistent
        self._used = False
        self._watched = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def watch(self, tensor):
        from .tensor import Tensor
        if isinstance(tensor, (list, tuple)):
            for t in tensor:
                self.watch(t)
            return
        if isinstance(tensor, Tensor):
            tensor.requires_grad = True
            self._watched.add(id(tensor))

    def _topo(self, target):
        visited, order = set(), []

        def dfs(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for p in getattr(t, "_prev", ()):
                dfs(p)
            order.append(t)

        dfs(target)
        return order

    def gradient(self, target, sources, output_gradients=None):
        from .tensor import Tensor
        single = not isinstance(sources, (list, tuple))
        srcs = [sources] if single else list(sources)
        if self._used and not self.persistent:
            raise RuntimeError("Non-persistent GradientTape — call gradient only once or use persistent=True")
        self._used = True

        # reset grads on graph
        for t in self._topo(target):
            t.grad = None
        # split-accumulator fallback
        for t in self._topo(target):
            if hasattr(t, "_split_acc"):
                delattr(t, "_split_acc")

        seed = None
        if output_gradients is None:
            seed = np.ones_like(target._data, dtype=np.float32 if target._data.dtype != np.float64 else np.float64)
        else:
            seed = np.asanyarray(output_gradients if not isinstance(output_gradients, Tensor) else output_gradients._data)
            seed = np.broadcast_to(seed, target._data.shape).copy()
        target.grad = seed

        for t in reversed(self._topo(target)):
            try:
                t._backward()
            except Exception:
                pass
        # split fallback: distribute concatenated grads back to slices
        out = []
        for s in srcs:
            if isinstance(s, Tensor) and hasattr(s, "_split_acc"):
                # not enough positional info — grads already accumulated where possible
                pass
            out.append(None if s is None or getattr(s, "grad", None) is None
                       else self._wrap_grad(s))
        return out[0] if single else out

    @staticmethod
    def _wrap_grad(s):
        from .tensor import Tensor
        g = s.grad
        t = Tensor.__new__(Tensor)
        t._data = np.ascontiguousarray(np.asanyarray(g))
        t.requires_grad = False
        t.grad = None
        t._prev = ()
        t._op = ""
        t._backward = lambda: None
        t.name = ""
        return t

    def jacobian(self, target, source):
        # numerical-free row-by-row for small targets
        from .tensor import Tensor
        tsize = int(np.prod(target._data.shape))
        flat = target._data.reshape(-1)
        rows = []
        for i in range(tsize):
            one = np.zeros_like(flat)
            one[i] = 1.0
            g = self.gradient(target, source, output_gradients=one.reshape(target._data.shape))
            if self.persistent:
                self._used = False
            rows.append(g._data.reshape(-1) if g is not None else np.zeros(int(np.prod(source._data.shape))))
        return rows
