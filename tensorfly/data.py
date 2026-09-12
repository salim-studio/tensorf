"""tensorfly.data — like tf.data.Dataset (fast with prefetch)."""
from __future__ import annotations

import threading
import queue
from typing import Callable

import numpy as np


class Dataset:
    def __init__(self, gen_fn, length=None):
        self._gen_fn = gen_fn
        self._length = length

    def __iter__(self):
        return self._gen_fn()

    def __len__(self):
        if self._length is None:
            raise TypeError("length unknown")
        return self._length

    # -- constructors --
    @staticmethod
    def from_tensor_slices(tensors) -> "Dataset":
        from .tensor import Tensor
        if isinstance(tensors, (list, tuple)):
            arrs = [(t._data if isinstance(t, Tensor) else np.asanyarray(t)) for t in tensors]
            n = len(arrs[0])
            def gen():
                for i in range(n):
                    yield tuple(a[i] for a in arrs)
            return Dataset(gen, n)
        a = tensors._data if isinstance(tensors, Tensor) else np.asanyarray(tensors)
        def gen():
            for i in range(len(a)):
                yield a[i]
        return Dataset(gen, len(a))

    @staticmethod
    def from_tensors(*arrays) -> "Dataset":
        def gen():
            yield tuple(arrays)
        return Dataset(gen, 1)

    @staticmethod
    def range(n) -> "Dataset":
        def gen():
            for i in range(n):
                yield i
        return Dataset(gen, n)

    # -- transforms --
    def map(self, fn: Callable, num_parallel_calls=None) -> "Dataset":
        parent = self._gen_fn
        def gen():
            for x in parent():
                yield fn(x)
        return Dataset(gen, self._length)

    def batch(self, batch_size: int, drop_remainder=False) -> "Dataset":
        parent = self._gen_fn
        n = None if self._length is None else (self._length // batch_size if drop_remainder
                                               else (self._length + batch_size - 1) // batch_size)
        def gen():
            buf = []
            for x in parent():
                buf.append(x)
                if len(buf) == batch_size:
                    yield _stack_batch(buf)
                    buf = []
            if buf and not drop_remainder:
                yield _stack_batch(buf)
        return Dataset(gen, n)

    def shuffle(self, buffer_size=1000, seed=None) -> "Dataset":
        parent = self._gen_fn
        rng = np.random.default_rng(seed)
        def gen():
            buf = []
            for x in parent():
                buf.append(x)
                if len(buf) >= buffer_size:
                    rng.shuffle(buf)
                    yield from buf
                    buf = []
            rng.shuffle(buf)
            yield from buf
        return Dataset(gen, self._length)

    def take(self, n) -> "Dataset":
        parent = self._gen_fn
        def gen():
            for i, x in enumerate(parent()):
                if i >= n:
                    break
                yield x
        return Dataset(gen, min(n, self._length) if self._length is not None else n)

    def repeat(self, count=None) -> "Dataset":
        parent = self._gen_fn
        def gen():
            c = 0
            while count is None or c < count:
                yield from parent()
                c += 1
        return Dataset(gen, None if count is None or self._length is None else self._length * count)

    def prefetch(self, buffer_size=2) -> "Dataset":
        parent = self._gen_fn
        def gen():
            q: queue.Queue = queue.Queue(maxsize=max(1, buffer_size))
            stop = object()
            def fill():
                try:
                    for x in parent():
                        q.put(x)
                finally:
                    q.put(stop)
            th = threading.Thread(target=fill, daemon=True)
            th.start()
            while True:
                x = q.get()
                if x is stop:
                    break
                yield x
        return Dataset(gen, self._length)


def _stack_batch(buf):
    first = buf[0]
    if isinstance(first, (tuple, list)):
        return tuple(np.stack([b[i] for b in buf]) for i in range(len(first)))
    return np.stack(buf)
