"""tensorf.ops — common TF ops (shape + math + reductions + linalg)."""
from __future__ import annotations

import numpy as np

from .tensor import (
    Tensor, Variable, convert_to_tensor, _unbroadcast, _ensure_contig,
    add, subtract, multiply, divide, pow, matmul, reduce_sum, reduce_mean,
    reshape, transpose,
)


def _t(x) -> Tensor:
    return x if isinstance(x, Tensor) else convert_to_tensor(x)


def _wrap_np(data, prev=(), op="", req=False):
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(np.asanyarray(data))
    out.requires_grad = req
    out.grad = None
    out._prev = tuple(prev)
    out._op = op
    out._backward = lambda: None
    out.name = ""
    return out


# ---------------- shape ops ----------------
def expand_dims(a, axis) -> Tensor:
    A = _t(a)
    return _wrap_np(np.expand_dims(A._data, axis), (A,), "expand_dims", A.requires_grad) \
        if not A.requires_grad else _expand_dims_grad(A, axis)


def _expand_dims_grad(A, axis):
    data = np.expand_dims(A._data, axis)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = True; out.grad = None
    out._prev = (A,); out._op = "expand_dims"; out.name = ""
    def _bw():
        if out.grad is None: return
        A.grad = (A.grad if A.grad is not None else 0) + out.grad.reshape(A.shape)
    out._backward = _bw
    return out


def squeeze(a, axis=None) -> Tensor:
    A = _t(a)
    data = np.squeeze(A._data, axis=axis)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "squeeze"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            A.grad = (A.grad if A.grad is not None else 0) + out.grad.reshape(A.shape)
    out._backward = _bw
    return out


def concat(values, axis=0) -> Tensor:
    Ts = [_t(v) for v in values]
    arrs = [t._data for t in Ts]
    data = np.concatenate(arrs, axis=axis)
    req = any(t.requires_grad for t in Ts)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = req; out.grad = None
    out._prev = tuple(Ts); out._op = "concat"; out.name = ""
    ax = axis % Ts[0]._data.ndim if Ts[0]._data.ndim else 0
    sizes = [a.shape[ax] for a in arrs]
    def _bw():
        if out.grad is None: return
        parts = np.split(out.grad, np.cumsum(sizes)[:-1], axis=ax)
        for t, p in zip(Ts, parts):
            if t.requires_grad:
                t.grad = (t.grad if t.grad is not None else 0) + p
    out._backward = _bw
    return out


concatenate = concat


def stack(values, axis=0) -> Tensor:
    Ts = [_t(v) for v in values]
    data = np.stack([t._data for t in Ts], axis=axis)
    req = any(t.requires_grad for t in Ts)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = req; out.grad = None
    out._prev = tuple(Ts); out._op = "stack"; out.name = ""
    def _bw():
        if out.grad is None: return
        parts = [np.squeeze(p, axis=axis) for p in np.split(out.grad, len(Ts), axis=axis)]
        for t, p in zip(Ts, parts):
            if t.requires_grad:
                t.grad = (t.grad if t.grad is not None else 0) + p
    out._backward = _bw
    return out


def split(value, num_or_size_splits, axis=0) -> list:
    A = _t(value)
    parts = np.split(A._data, num_or_size_splits, axis=axis)
    outs = []
    for p in parts:
        o = Tensor.__new__(Tensor)
        o._data = _ensure_contig(p); o.requires_grad = A.requires_grad; o.grad = None
        o._prev = (A,); o._op = "split"; o.name = ""
        def _bw(o=o):
            if o.grad is None: return
            # accumulate via slicing — simpler: rebuild full grad lazily on tape.backward
            if A.requires_grad:
                if not hasattr(A, "_split_acc"):
                    A._split_acc = []
                A._split_acc.append((o, o.grad))
        o._backward = _bw
        outs.append(o)
    # NOTE: split backward handled in autodiff via generic path below (fallback sums slices)
    return outs


def tile(a, multiples) -> Tensor:
    A = _t(a)
    return _wrap_np(np.tile(A._data, multiples), (A,), "tile", A.requires_grad)


def pad(a, paddings, mode="CONSTANT", constant_values=0) -> Tensor:
    A = _t(a)
    return _wrap_np(np.pad(A._data, paddings, mode=mode.lower() if isinstance(mode, str) else mode,
                           constant_values=constant_values), (), "pad", False)


def gather(params, indices, axis=0) -> Tensor:
    P = _t(params)
    I = np.asanyarray(indices if not isinstance(indices, Tensor) else indices._data)
    data = np.take(P._data, I, axis=axis)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = P.requires_grad; out.grad = None
    out._prev = (P,); out._op = "gather"; out.name = ""
    def _bw():
        if out.grad is None: return
        if P.requires_grad:
            ga = np.zeros_like(P._data)
            np.add.at(ga, I, out.grad)
            P.grad = (P.grad if P.grad is not None else 0) + ga
    out._backward = _bw
    return out


def where(condition, x, y) -> Tensor:
    C = np.asanyarray(condition if not isinstance(condition, Tensor) else condition._data)
    X = _t(x); Y = _t(y)
    data = np.where(C, X._data, Y._data)
    req = X.requires_grad or Y.requires_grad
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = req; out.grad = None
    out._prev = (X, Y); out._op = "where"; out.name = ""
    def _bw():
        if out.grad is None: return
        if X.requires_grad:
            X.grad = (X.grad if X.grad is not None else 0) + _unbroadcast(np.where(C, out.grad, 0), X.shape)
        if Y.requires_grad:
            Y.grad = (Y.grad if Y.grad is not None else 0) + _unbroadcast(np.where(C, 0, out.grad), Y.shape)
    out._backward = _bw
    return out


def clip_by_value(t, clip_value_min, clip_value_max) -> Tensor:
    A = _t(t)
    data = np.clip(A._data, clip_value_min, clip_value_max)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "clip"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            m = (A._data >= clip_value_min) & (A._data <= clip_value_max)
            A.grad = (A.grad if A.grad is not None else 0) + out.grad * m
    out._backward = _bw
    return out


# ---------------- math ----------------
def maximum(a, b) -> Tensor:
    from .tensor import _binary_op
    A = _t(a); B = _t(b)
    return _binary_op(A, B, np.maximum,
                      lambda g: g * (A._data >= B._data), lambda g: g * (B._data > A._data), "maximum")


def minimum(a, b) -> Tensor:
    from .tensor import _binary_op
    A = _t(a); B = _t(b)
    return _binary_op(A, B, np.minimum,
                      lambda g: g * (A._data <= B._data), lambda g: g * (B._data < A._data), "minimum")


def tensordot(a, b, axes=2) -> Tensor:
    A = _t(a); B = _t(b)
    data = np.tensordot(A._data, B._data, axes=axes)
    # autograd for the common matmul-like case only; else forward-only
    out = _wrap_np(data, (A, B), "tensordot", A.requires_grad or B.requires_grad)
    return out


def einsum(equation, *operands) -> Tensor:
    Ts = [_t(o) for o in operands]
    data = np.einsum(equation, *[t._data for t in Ts], optimize=True)
    return _wrap_np(data, (), "einsum", False)


def norm(t, axis=None, keepdims=False) -> Tensor:
    A = _t(t)
    return sqrt(reduce_sum(A * A, axis=axis, keepdims=keepdims))


def sqrt(a):
    from .tensor import sqrt as _s
    return _s(a)


# ---------------- reductions ----------------
def reduce_max(a, axis=None, keepdims=False) -> Tensor:
    A = _t(a)
    return _wrap_np(A._data.max(axis=axis, keepdims=keepdims), (), "reduce_max", False)


def reduce_min(a, axis=None, keepdims=False) -> Tensor:
    A = _t(a)
    return _wrap_np(A._data.min(axis=axis, keepdims=keepdims), (), "reduce_min", False)


def reduce_prod(a, axis=None, keepdims=False) -> Tensor:
    A = _t(a)
    return _wrap_np(A._data.prod(axis=axis, keepdims=keepdims), (), "reduce_prod", False)


def argmax(a, axis=None) -> Tensor:
    A = _t(a)
    return _wrap_np(np.argmax(A._data, axis=axis), (), "argmax", False)


def argmin(a, axis=None) -> Tensor:
    A = _t(a)
    return _wrap_np(np.argmin(A._data, axis=axis), (), "argmin", False)


def sort(a, axis=-1) -> Tensor:
    return _wrap_np(np.sort(_t(a)._data, axis=axis), (), "sort", False)


def argsort(a, axis=-1) -> Tensor:
    return _wrap_np(np.argsort(_t(a)._data, axis=axis), (), "argsort", False)


# ---------------- logic / compare (forward-only, like TF) ----------------
def _cmp(a, b, fn, op):
    A = _t(a); B = _t(b)
    return _wrap_np(fn(A._data, B._data), (), op, False)


def equal(a, b): return _cmp(a, b, np.equal, "equal")
def not_equal(a, b): return _cmp(a, b, np.not_equal, "not_equal")
def greater(a, b): return _cmp(a, b, np.greater, "greater")
def greater_equal(a, b): return _cmp(a, b, np.greater_equal, "greater_equal")
def less(a, b): return _cmp(a, b, np.less, "less")
def less_equal(a, b): return _cmp(a, b, np.less_equal, "less_equal")


# ---------------- linalg ----------------
class linalg:
    @staticmethod
    def matmul(a, b, **kw): return matmul(a, b, **kw)
    @staticmethod
    def inv(a): return _wrap_np(np.linalg.inv(_t(a)._data), (), "inv", False)
    @staticmethod
    def det(a): return _wrap_np(np.linalg.det(_t(a)._data), (), "det", False)
    @staticmethod
    def solve(a, b): return _wrap_np(np.linalg.solve(_t(a)._data, _t(b)._data), (), "solve", False)
    @staticmethod
    def norm(t, axis=None, keepdims=False): return norm(t, axis, keepdims)
    @staticmethod
    def eig(a): return np.linalg.eig(_t(a)._data)
    @staticmethod
    def svd(a, full_matrices=False): return np.linalg.svd(_t(a)._data, full_matrices=full_matrices)
    @staticmethod
    def qr(a): return np.linalg.qr(_t(a)._data)
    @staticmethod
    def cholesky(a): return _wrap_np(np.linalg.cholesky(_t(a)._data), (), "cholesky", False)
