"""tensorfly.parallel — acceleration engine (multithreading + fusion).

Speed sources vs tensorflow (CPU):
1. Chunked ThreadPool for element-wise ops — numpy is single-threaded there.
2. Parallel batch-matmul over the batch dimension.
3. Fused ops (dense+bias+act, adam-step) to cut temporaries.
4. Contiguous memory + float32 by default.
5. Fully vectorized im2col for Conv2D.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MAX_WORKERS = max(1, (os.cpu_count() or 4))
PARALLEL_THRESHOLD = 1 << 18  # 262k elements

_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    return _executor


def get_workers() -> int:
    return MAX_WORKERS


def set_workers(n: int) -> None:
    global MAX_WORKERS, _executor
    MAX_WORKERS = max(1, int(n))
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def _chunks(n: int, w: int):
    base = n // w
    rem = n % w
    s = 0
    for i in range(w):
        e = s + base + (1 if i < rem else 0)
        if e > s:
            yield s, e
        s = e


def parallel_ewise(func, *arrays, out=None, workers: int | None = None):
    """Apply func(*views, out=out_view) over contiguous chunks in parallel.

    func receives flattened views. Returns `out` (or new array shaped like
    broadcast result — caller must ensure broadcast already resolved).
    """
    w = workers or MAX_WORKERS
    # resolve broadcast shape
    arrs = [np.asanyarray(a) for a in arrays]
    shape = np.broadcast_shapes(*[a.shape for a in arrs]) if arrs else ()
    flats = [np.broadcast_to(a, shape).ravel() for a in arrs]
    if out is None:
        out = np.empty(shape, dtype=np.result_type(*[a.dtype for a in flats]) if flats else np.float32)
    out_flat = out.ravel()
    n = out_flat.size
    if n < PARALLEL_THRESHOLD or w <= 1:
        func(*flats, out=out_flat)
        return out
    ex = get_executor()
    views = [f[s:e] for f in flats for s, e in []]  # placeholder
    futs = []
    for s, e in _chunks(n, min(w, 8)):
        vs = [f[s:e] for f in flats]
        o = out_flat[s:e]
        futs.append(ex.submit(func, *vs, out=o))
    for f in futs:
        f.result()
    return out


def parallel_batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (..., M, K), B: (..., K, N). BLAS already batches optimally for
    large matrices — parallelize only many-small-matrix batches."""
    if A.ndim <= 2 and B.ndim <= 2:
        return A @ B
    m, k, n = A.shape[-2], A.shape[-1], B.shape[-1]
    batch = int(np.prod(A.shape[:-2])) if A.ndim > 2 else 1
    # BLAS wins for medium/large matrices; threading only helps many tiny ones
    if max(m, k, n) >= 64 or batch < 8:
        return A @ B
    A2 = A.reshape(batch, A.shape[-2], A.shape[-1])
    # B may broadcast
    if B.ndim == 2:
        return A @ B
    B2 = np.broadcast_to(B, A.shape[:-2] + (B.shape[-2], B.shape[-1])).reshape(batch, B.shape[-2], B.shape[-1])
    ex = get_executor()
    out = np.empty((batch, A.shape[-2], B.shape[-1]), dtype=np.result_type(A.dtype, B.dtype))
    w = min(MAX_WORKERS, batch)

    def _job(idxs):
        for i in idxs:
            out[i] = A2[i] @ B2[i]

    idx = list(range(batch))
    groups = [idx[i::w] for i in range(w)]
    list(ex.map(_job, groups))
    return out.reshape(A.shape[:-2] + (A.shape[-2], B.shape[-1]))
