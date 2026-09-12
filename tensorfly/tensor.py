"""tensorfly.tensor — Tensor + Variable core and creation ops (like tf.*)."""
from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

from .parallel import parallel_ewise, parallel_batch_matmul, PARALLEL_THRESHOLD
from .dtypes import as_dtype


# ---------------------------------------------------------------- helpers
def _to_np(x) -> np.ndarray:
    if isinstance(x, Tensor):
        return x._data
    return np.asanyarray(x)


def _ensure_contig(a: np.ndarray, dtype=None) -> np.ndarray:
    if dtype is not None:
        a = a.astype(dtype, copy=False)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad down to `shape` (inverse of numpy broadcasting)."""
    if grad.shape == tuple(shape):
        return grad
    # leading extra dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
        elif g != s:
            # broadcast dim (s missing -> treated as 1)
            grad = grad.sum(axis=i, keepdims=True)
    if grad.shape != tuple(shape):
        grad = grad.reshape(shape)
    return grad


# ---------------------------------------------------------------- Tensor
class Tensor:
    """TF-like array with autograd (eager by default, like TF2)."""

    __slots__ = ("_data", "requires_grad", "grad", "_prev", "_op", "_backward", "name")

    def __init__(self, data, dtype=None, requires_grad: bool = False, name: str = ""):
        dt = as_dtype(dtype)
        arr = _to_np(data)
        if dt is not None:
            arr = arr.astype(dt, copy=False)
        # default float32 like TF (numpy defaults float64 — we prefer float32 for speed)
        if arr.dtype == np.float64 and dt is None:
            arr = arr.astype(np.float32, copy=False)
        self._data: np.ndarray = _ensure_contig(np.asanyarray(arr))
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._prev: tuple = ()
        self._op: str = ""
        self._backward: Callable[[], None] = lambda: None
        self.name = name

    # -- TF/Torch compat properties --
    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def size(self):
        return self._data.size

    @property
    def data(self):
        return self._data

    def numpy(self) -> np.ndarray:
        return self._data

    def detach(self) -> "Tensor":
        return Tensor(self._data, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    # -- dunder (so `a + b`, `a @ b`, `a * 2` work and stay differentiable) --
    def _wrap_out(self, data, prev, op, backward, requires_grad):
        t = Tensor.__new__(Tensor)
        t._data = _ensure_contig(np.asanyarray(data))
        t.requires_grad = requires_grad
        t.grad = None
        t._prev = prev
        t._op = op
        t._backward = backward
        t.name = ""
        return t

    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(o, self)
    def __sub__(self, o): return subtract(self, o)
    def __rsub__(self, o): return subtract(o, self)
    def __mul__(self, o): return multiply(self, o)
    def __rmul__(self, o): return multiply(o, self)
    def __truediv__(self, o): return divide(self, o)
    def __rtruediv__(self, o): return divide(o, self)
    def __pow__(self, o): return pow(self, o)
    def __neg__(self): return negative(self)
    def __matmul__(self, o): return matmul(self, o)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return _slice(self, idx)

    def __setitem__(self, idx, value):
        # TF Variables support assign via __setitem__; keep graph-free (like .assign region)
        v = _to_np(value)
        self._data[idx] = v

    def __array__(self, dtype=None):
        return self._data.astype(dtype) if dtype else self._data

    def __float__(self):
        return float(self._data)

    def __repr__(self):
        return f"tensorfly.Tensor(shape={self.shape}, dtype={self.dtype})\\n{self._data}"

    # TF-style helpers
    def assign(self, value):
        v = _to_np(value)
        if v.shape != self.shape:
            # allow reshape-assign like tf.Variable.assign with same size
            v = np.broadcast_to(v, self.shape).copy()
        np.copyto(self._data, v)
        return self

    def assign_add(self, delta):
        np.add(self._data, _to_np(delta), out=self._data)
        return self

    def assign_sub(self, delta):
        np.subtract(self._data, _to_np(delta), out=self._data)
        return self


class Variable(Tensor):
    """tf.Variable — always trainable (requires_grad=True)."""

    def __init__(self, initial_value=None, dtype=None, name: str = "", trainable: bool = True):
        super().__init__([] if initial_value is None else initial_value,
                         dtype=dtype, requires_grad=bool(trainable), name=name)

    @property
    def trainable(self):
        return self.requires_grad


# ---------------------------------------------------------------- creation ops (tf.*)
def constant(value, dtype=None, shape=None, name="") -> Tensor:
    a = _to_np(value)
    if shape is not None:
        a = np.broadcast_to(a, tuple(shape)).copy()
    return Tensor(a, dtype=dtype, requires_grad=False, name=name)


def convert_to_tensor(value, dtype=None) -> Tensor:
    if isinstance(value, Tensor):
        if dtype is None or np.dtype(value.dtype) == as_dtype(dtype):
            return value
        return Tensor(value._data, dtype=dtype)
    return constant(value, dtype=dtype)


def zeros(shape, dtype=np.float32) -> Tensor:
    return Tensor(np.zeros(tuple(shape) if isinstance(shape, (list, tuple)) else shape, dtype=as_dtype(dtype) or np.float32))


def ones(shape, dtype=np.float32) -> Tensor:
    return Tensor(np.ones(tuple(shape) if isinstance(shape, (list, tuple)) else shape, dtype=as_dtype(dtype) or np.float32))


def zeros_like(t, dtype=None) -> Tensor:
    a = _to_np(t)
    return Tensor(np.zeros_like(a, dtype=as_dtype(dtype) or a.dtype))


def ones_like(t, dtype=None) -> Tensor:
    a = _to_np(t)
    return Tensor(np.ones_like(a, dtype=as_dtype(dtype) or a.dtype))


def fill(dims, value) -> Tensor:
    v = _to_np(value)
    shape = tuple(dims) if isinstance(dims, (list, tuple)) else (dims,)
    dt = v.dtype if v.size else np.float32
    return Tensor(np.full(shape, v.flat[0] if v.size else 0, dtype=dt))


def eye(n, m=None, dtype=np.float32) -> Tensor:
    return Tensor(np.eye(n, m, dtype=as_dtype(dtype) or np.float32))


def range(*args, dtype=None) -> Tensor:
    r = np.arange(*args, dtype=as_dtype(dtype) or np.int32)
    return Tensor(r)


def linspace(start, stop, num, dtype=np.float32) -> Tensor:
    return Tensor(np.linspace(start, stop, num, dtype=as_dtype(dtype) or np.float32))


def cast(t, dtype) -> Tensor:
    return Tensor(_to_np(t).astype(as_dtype(dtype)), requires_grad=False)


def one_hot(indices, depth, on_value=1.0, off_value=0.0, dtype=np.float32) -> Tensor:
    idx = _to_np(indices).astype(np.int64).ravel()
    dt = as_dtype(dtype) or np.float32
    out = np.full((idx.size, depth), off_value, dtype=dt)
    mask = (idx >= 0) & (idx < depth)
    out[np.arange(idx.size)[mask], idx[mask]] = on_value
    return Tensor(out.reshape(_to_np(indices).shape + (depth,)))


def _fast_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size >= PARALLEL_THRESHOLD or b.size >= PARALLEL_THRESHOLD:
        shape = np.broadcast_shapes(a.shape, b.shape)
        out = np.empty(shape, dtype=np.result_type(a.dtype, b.dtype))
        fa = np.broadcast_to(a, shape).ravel()
        fb = np.broadcast_to(b, shape).ravel()
        parallel_ewise(lambda x, y, out: np.add(x, y, out=out), fa, fb, out=out.ravel())
        return out
    return np.add(a, b)


# ---------------------------------------------------------------- differentiable elementwise
def _binary_op(a, b, fwd, bwd_a, bwd_b, op):
    A = a if isinstance(a, Tensor) else constant(a)
    B = b if isinstance(b, Tensor) else constant(b)
    data = fwd(A._data, B._data)
    req = A.requires_grad or B.requires_grad
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(np.asanyarray(data))
    out.requires_grad = req
    out.grad = None
    out._prev = (A, B)
    out._op = op
    out.name = ""

    def _bw():
        if out.grad is None:
            return
        g = out.grad
        if A.requires_grad:
            ga = _unbroadcast(bwd_a(g), A.shape)
            A.grad = ga if A.grad is None else A.grad + ga
        if B.requires_grad:
            gb = _unbroadcast(bwd_b(g), B.shape)
            B.grad = gb if B.grad is None else B.grad + gb

    out._backward = _bw
    return out


def add(a, b) -> Tensor:
    return _binary_op(a, b, _fast_add, lambda g: g, lambda g: g, "add")


def subtract(a, b) -> Tensor:
    return _binary_op(a, b, np.subtract, lambda g: g, lambda g: -g, "sub")


def multiply(a, b) -> Tensor:
    A = a if isinstance(a, Tensor) else constant(a)
    B = b if isinstance(b, Tensor) else constant(b)
    return _binary_op(A, B, np.multiply, lambda g: g * B._data, lambda g: g * A._data, "mul")


def divide(a, b) -> Tensor:
    A = a if isinstance(a, Tensor) else constant(a)
    B = b if isinstance(b, Tensor) else constant(b)
    return _binary_op(A, B, np.divide, lambda g: g / B._data, lambda g: -g * A._data / (B._data ** 2), "div")


def pow(a, b) -> Tensor:
    A = a if isinstance(a, Tensor) else constant(a)
    if isinstance(b, Tensor):
        B = b
        data = np.power(A._data, B._data)
        req = A.requires_grad or B.requires_grad
        out = Tensor.__new__(Tensor)
        out._data = _ensure_contig(data); out.requires_grad = req; out.grad = None
        out._prev = (A, B); out._op = "pow"; out.name = ""
        def _bw():
            if out.grad is None: return
            g = out.grad
            if A.requires_grad:
                A.grad = (A.grad if A.grad is not None else 0) + _unbroadcast(g * B._data * np.power(A._data, B._data - 1, where=(A._data != 0)), A.shape)
            if B.requires_grad:
                B.grad = (B.grad if B.grad is not None else 0) + _unbroadcast(g * data * np.log(np.maximum(A._data, 1e-12)), B.shape)
        out._backward = _bw
        return out
    data = np.power(A._data, b)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "pow"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            A.grad = (A.grad if A.grad is not None else 0) + _unbroadcast(out.grad * b * np.power(A._data, b - 1, where=(A._data != 0)), A.shape)
    out._backward = _bw
    return out


def negative(a) -> Tensor:
    A = convert_to_tensor(a)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(-A._data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "neg"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            A.grad = (A.grad if A.grad is not None else 0) + (-out.grad)
    out._backward = _bw
    return out


def _unary_op(a, fwd, bwd, op):
    A = convert_to_tensor(a)
    data = fwd(A._data)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(np.asanyarray(data)); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = op; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            A.grad = (A.grad if A.grad is not None else 0) + _unbroadcast(bwd(out.grad, A._data, out._data), A.shape)
    out._backward = _bw
    return out


def exp(a): return _unary_op(a, np.exp, lambda g, x, y: g * y, "exp")
def log(a): return _unary_op(a, np.log, lambda g, x, y: g / np.maximum(x, 1e-12), "log")
def sqrt(a): return _unary_op(a, np.sqrt, lambda g, x, y: g / (2 * np.maximum(y, 1e-12)), "sqrt")
def square(a): return _unary_op(a, np.square, lambda g, x, y: g * 2 * x, "square")
def abs_(a): return _unary_op(a, np.abs, lambda g, x, y: g * np.sign(x), "abs")


def matmul(a, b, transpose_a=False, transpose_b=False) -> Tensor:
    A = convert_to_tensor(a)
    B = convert_to_tensor(b)
    Ad = A._data.T if transpose_a and A.ndim == 2 else (np.swapaxes(A._data, -1, -2) if transpose_a else A._data)
    Bd = B._data.T if transpose_b and B.ndim == 2 else (np.swapaxes(B._data, -1, -2) if transpose_b else B._data)
    data = Ad @ Bd if Ad.ndim <= 2 and Bd.ndim <= 2 else parallel_batch_matmul(Ad, Bd)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad or B.requires_grad; out.grad = None
    out._prev = (A, B); out._op = "matmul"; out.name = ""
    def _bw():
        if out.grad is None: return
        g = out.grad
        if A.requires_grad:
            if Bd.ndim <= 2:
                ga = g @ Bd.T
            else:
                ga = parallel_batch_matmul(g, np.swapaxes(Bd, -1, -2))
            if transpose_a:
                ga = np.swapaxes(ga, -1, -2) if ga.ndim > 2 else ga.T
            A.grad = (A.grad if A.grad is not None else 0) + _unbroadcast(ga, A.shape)
        if B.requires_grad:
            if Ad.ndim <= 2:
                gb = Ad.T @ g
            else:
                gb = parallel_batch_matmul(np.swapaxes(Ad, -1, -2), g)
            if transpose_b:
                gb = np.swapaxes(gb, -1, -2) if gb.ndim > 2 else gb.T
            B.grad = (B.grad if B.grad is not None else 0) + _unbroadcast(gb, B.shape)
    out._backward = _bw
    return out


def reduce_sum(a, axis=None, keepdims=False) -> Tensor:
    A = convert_to_tensor(a)
    data = A._data.sum(axis=axis, keepdims=keepdims)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(np.asanyarray(data)); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "reduce_sum"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            g = out.grad
            if not keepdims and axis is not None:
                ax = (axis,) if isinstance(axis, int) else tuple(axis)
                shape = list(A.shape)
                for i in ax:
                    shape[i] = 1
                g = g.reshape(shape)
            A.grad = (A.grad if A.grad is not None else 0) + np.broadcast_to(g, A.shape).copy()
    out._backward = _bw
    return out


def reduce_mean(a, axis=None, keepdims=False) -> Tensor:
    A = convert_to_tensor(a)
    data = A._data.mean(axis=axis, keepdims=keepdims)
    n = A._data.size / np.asanyarray(data).size
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(np.asanyarray(data)); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "reduce_mean"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            g = out.grad
            if not keepdims and axis is not None:
                ax = (axis,) if isinstance(axis, int) else tuple(axis)
                shape = list(A.shape)
                for i in ax:
                    shape[i] = 1
                g = g.reshape(shape)
            A.grad = (A.grad if A.grad is not None else 0) + (np.broadcast_to(g, A.shape) / n).copy()
    out._backward = _bw
    return out


def _slice(a, idx) -> Tensor:
    A = convert_to_tensor(a)
    data = A._data[idx]
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(np.asanyarray(data)); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "slice"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            ga = np.zeros_like(A._data)
            try:
                # works for slices / integer / fancy indexing
                tmp = np.zeros_like(A._data)
                np.add.at(tmp, idx, out.grad)
                ga = tmp
            except Exception:
                ga[idx] += out.grad
            A.grad = (A.grad if A.grad is not None else 0) + ga
    out._backward = _bw
    return out


def reshape(a, shape) -> Tensor:
    A = convert_to_tensor(a)
    data = A._data.reshape(shape)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "reshape"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            A.grad = (A.grad if A.grad is not None else 0) + out.grad.reshape(A.shape)
    out._backward = _bw
    return out


def transpose(a, perm=None) -> Tensor:
    A = convert_to_tensor(a)
    data = np.transpose(A._data, perm)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "transpose"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            if perm is None:
                A.grad = (A.grad if A.grad is not None else 0) + np.transpose(out.grad)
            else:
                inv = np.argsort(perm)
                A.grad = (A.grad if A.grad is not None else 0) + np.transpose(out.grad, inv)
    out._backward = _bw
    return out
