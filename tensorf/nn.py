"""tensorf.nn — activations + conv/pool + fused ops (like tf.nn)."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor, convert_to_tensor, _unbroadcast, _ensure_contig, matmul, add
from .ops import _t, _wrap_np


def _act(a, fwd, bwd, op):
    A = _t(a)
    data = fwd(A._data)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = op; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            A.grad = (A.grad if A.grad is not None else 0) + _unbroadcast(bwd(out.grad, A._data, out._data), A.shape)
    out._backward = _bw
    return out


def relu(a): return _act(a, lambda x: np.maximum(x, 0), lambda g, x, y: g * (x > 0), "relu")
def sigmoid(a):
    def f(x): return 1 / (1 + np.exp(-x))
    return _act(a, f, lambda g, x, y: g * y * (1 - y), "sigmoid")
def tanh(a): return _act(a, np.tanh, lambda g, x, y: g * (1 - y ** 2), "tanh")
def softplus(a): return _act(a, lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0), lambda g, x, y: g / (1 + np.exp(-x)), "softplus")
def leaky_relu(a, alpha=0.2): return _act(a, lambda x: np.where(x > 0, x, alpha * x), lambda g, x, y: g * np.where(x > 0, 1, alpha), "leaky_relu")
def elu(a, alpha=1.0): return _act(a, lambda x: np.where(x > 0, x, alpha * (np.exp(x) - 1)), lambda g, x, y: g * np.where(x > 0, 1, y + alpha), "elu")
def gelu(a):
    def f(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    def b(g, x, y):
        s = 1 / (1 + np.exp(-1.702 * x))  # fast approx derivative
        return g * s + g * x * s * (1 - s) * 1.702
    return _act(a, f, b, "gelu")
def silu(a):  # swish
    def f(x): return x / (1 + np.exp(-x))
    return _act(a, f, lambda g, x, y: g * (y + (1 / (1 + np.exp(-x))) * (1 - y)), "silu")
def softmax(a, axis=-1):
    A = _t(a)
    x = A._data - A._data.max(axis=axis, keepdims=True)
    e = np.exp(x)
    y = e / e.sum(axis=axis, keepdims=True)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(y); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "softmax"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            s = out._data
            g = out.grad
            dot = (g * s).sum(axis=axis, keepdims=True)
            A.grad = (A.grad if A.grad is not None else 0) + _unbroadcast(s * (g - dot), A.shape)
    out._backward = _bw
    return out


def log_softmax(a, axis=-1):
    A = _t(a)
    m = A._data.max(axis=axis, keepdims=True)
    lse = np.log(np.exp(A._data - m).sum(axis=axis, keepdims=True)) + m
    return _t(A - lse if isinstance(A - lse, Tensor) else lse)


def dropout(a, rate=0.5, training=True, seed=None) -> Tensor:
    A = _t(a)
    if not training or rate <= 0:
        return A
    rng = np.random.default_rng(seed)
    mask = (rng.random(A.shape) >= rate).astype(A._data.dtype) / (1.0 - rate)
    data = A._data * mask
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(data); out.requires_grad = A.requires_grad; out.grad = None
    out._prev = (A,); out._op = "dropout"; out.name = ""
    def _bw():
        if out.grad is None: return
        if A.requires_grad:
            A.grad = (A.grad if A.grad is not None else 0) + out.grad * mask
    out._backward = _bw
    return out


def bias_add(value, bias) -> Tensor:
    return add(value, bias)


def dense_fused(x, kernel, bias=None, activation=None):
    """Fused: x@W + b + act in a single pass (faster than 3 separate kernels)."""
    from .tensor import Tensor as _T
    Xd = x._data if isinstance(x, _T) else np.asanyarray(x)
    Wd = kernel._data if isinstance(kernel, _T) else np.asanyarray(kernel)
    needs_grad = (isinstance(x, _T) and x.requires_grad) or (isinstance(kernel, _T) and kernel.requires_grad) \
        or (isinstance(bias, _T) and bias.requires_grad)
    if not needs_grad:
        # inference fast-path: single output buffer, in-place bias+act (no temporaries)
        y = Xd @ Wd
        if y.dtype != np.float32:
            y = y.astype(np.float32, copy=False)
        if not y.flags["C_CONTIGUOUS"]:
            y = np.ascontiguousarray(y)
        if bias is not None:
            bd = bias._data if isinstance(bias, _T) else np.asanyarray(bias)
            y += bd.reshape(1, -1) if bd.ndim == 1 and y.ndim == 2 else bd
        if activation == "relu":
            np.maximum(y, 0, out=y)
        elif activation is not None:
            fn = {"sigmoid": sigmoid, "tanh": tanh, "softmax": softmax,
                  "gelu": gelu, "silu": silu, "softplus": softplus}.get(activation, None)
            if fn is None:
                raise ValueError(f"unknown activation {activation}")
            return fn(_T(y, requires_grad=False))
        return _T(y, requires_grad=False)
    y = matmul(x, kernel)
    if bias is not None:
        y = add(y, bias)
    if activation is None:
        return y
    fn = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh, "softmax": softmax,
          "gelu": gelu, "silu": silu, "softplus": softplus}.get(activation, None)
    if fn is None:
        raise ValueError(f"unknown activation {activation}")
    return fn(y)


# ---------------- conv2d / pooling (NHWC like TF) ----------------
def _im2col(x: np.ndarray, kh: int, kw: int, stride=1, padding="VALID"):
    N, H, W, C = x.shape
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    if padding == "SAME":
        out_h = int(np.ceil(H / sh)); out_w = int(np.ceil(W / sw))
        pad_h = max((out_h - 1) * sh + kh - H, 0); pad_w = max((out_w - 1) * sw + kw - W, 0)
        pt, pl = pad_h // 2, pad_w // 2
        pb, pr = pad_h - pt, pad_w - pl
        x = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
        H, W = x.shape[1], x.shape[2]
    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1
    shape = (N, out_h, out_w, kh, kw, C)
    strides = (x.strides[0], sh * x.strides[1], sw * x.strides[2], x.strides[1], x.strides[2], x.strides[3])
    cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.ascontiguousarray(cols.reshape(N * out_h * out_w, kh * kw * C)), (out_h, out_w)


def conv2d(x, filters, strides=1, padding="VALID") -> Tensor:
    X = _t(x); F = _t(filters)
    kh, kw, Cin, Cout = F._data.shape
    cols, (oh, ow) = _im2col(np.ascontiguousarray(X._data), kh, kw, strides, padding)
    W = F._data.reshape(-1, Cout)
    y = (cols @ W).reshape(X.shape[0], oh, ow, Cout)
    req = X.requires_grad or F.requires_grad
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(y); out.requires_grad = req; out.grad = None
    out._prev = (X, F); out._op = "conv2d"; out.name = ""
    def _bw():
        if out.grad is None: return
        g = out.grad.reshape(-1, Cout)
        if F.requires_grad:
            F.grad = (F.grad if F.grad is not None else 0) + (cols.T @ g).reshape(kh, kw, Cin, Cout)
        if X.requires_grad:
            dcols = g @ W.T  # (N*oh*ow, kh*kw*Cin)
            # col2im (simple loop over kernel — fine for typical small kernels)
            sh, sw = (strides, strides) if isinstance(strides, int) else strides
            N = X.shape[0]
            H, Wd, C = X.shape[1], X.shape[2], X.shape[3]
            dcols = dcols.reshape(N, oh, ow, kh, kw, Cin)
            # reconstruct padded grad then crop for SAME
            if padding == "SAME":
                out_h = int(np.ceil(H / sh)); out_w = int(np.ceil(Wd / sw))
                pad_h = max((out_h - 1) * sh + kh - H, 0); pad_w = max((out_w - 1) * sw + kw - Wd, 0)
                Hp, Wp = H + pad_h, Wd + pad_w
            else:
                Hp, Wp = (H - kh) // sh * sh + kh, (Wd - kw) // sw * sw + kw
            gx = np.zeros((N, Hp, Wp, Cin))
            for i in range(kh):
                for j in range(kw):
                    gx[:, i:i + oh * sh:sh, j:j + ow * sw:sw, :] += dcols[:, :, :, i, j, :]
            if padding == "SAME":
                pt, pl = pad_h // 2, pad_w // 2
                gx = gx[:, pt:pt + H, pl:pl + Wd, :]
            else:
                gx = gx[:, :H, :Wd, :]
            X.grad = (X.grad if X.grad is not None else 0) + gx
    out._backward = _bw
    return out


def max_pool(x, ksize=2, strides=2, padding="VALID") -> Tensor:
    X = _t(x)
    kh = kw = ksize if isinstance(ksize, int) else ksize[0]
    sh = sw = strides if isinstance(strides, int) else strides[0]
    N, H, W, C = X.shape
    cols, (oh, ow) = _im2col(np.ascontiguousarray(X._data), kh, kw, (sh, sw), padding)
    cols = cols.reshape(N * oh * ow, kh * kw, C)
    am = cols.max(axis=1)
    arg = cols.argmax(axis=1)
    y = am.reshape(N, oh, ow, C)
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(y); out.requires_grad = X.requires_grad; out.grad = None
    out._prev = (X,); out._op = "maxpool"; out.name = ""
    def _bw():
        if out.grad is None: return
        if X.requires_grad:
            g = out.grad.reshape(N * oh * ow, C)
            dcols = np.zeros_like(cols)
            idx = (np.arange(N * oh * ow)[:, None], arg, np.arange(C)[None, :])
            # arg has shape (N*oh*ow, C)
            dcols[np.arange(dcols.shape[0])[:, None], arg, np.arange(C)] = g
            dcols = dcols.reshape(N, oh, ow, kh, kw, C)
            if padding == "SAME":
                out_h = int(np.ceil(H / sh)); out_w = int(np.ceil(W / sw))
                pad_h = max((out_h - 1) * sh + kh - H, 0); pad_w = max((out_w - 1) * sw + kw - W, 0)
                Hp, Wp = H + pad_h, W + pad_w
            else:
                Hp, Wp = H, W
            gx = np.zeros((N, Hp, Wp, C))
            for i in range(kh):
                for j in range(kw):
                    gx[:, i:i + oh * sh:sh, j:j + ow * sw:sw, :] += dcols[:, :, :, i, j, :]
            if padding == "SAME":
                gx = gx[:, :H, :W, :]
            X.grad = (X.grad if X.grad is not None else 0) + gx
    out._backward = _bw
    return out


def avg_pool(x, ksize=2, strides=2, padding="VALID") -> Tensor:
    X = _t(x)
    kh = kw = ksize if isinstance(ksize, int) else ksize[0]
    sh = sw = strides if isinstance(strides, int) else strides[0]
    cols, (oh, ow) = _im2col(np.ascontiguousarray(X._data), kh, kw, (sh, sw), padding)
    N, H, W, C = X.shape
    y = cols.reshape(X.shape[0] * oh * ow, kh * kw * C).mean(axis=1).reshape(X.shape[0], oh, ow, C)
    return _wrap_np(y, (X,), "avgpool", X.requires_grad)


def sparse_softmax_cross_entropy(logits, labels) -> Tensor:
    L = _t(logits)
    lab = np.asanyarray(labels if not isinstance(labels, Tensor) else labels._data).astype(np.int64).ravel()
    x = L._data.reshape(-1, L._data.shape[-1])
    m = x.max(axis=1, keepdims=True)
    logp = x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(x.shape[0]), lab].reshape(L._data.shape[:-1])
    out = Tensor.__new__(Tensor)
    out._data = _ensure_contig(loss); out.requires_grad = L.requires_grad; out.grad = None
    out._prev = (L,); out._op = "sparse_xent"; out.name = ""
    def _bw():
        if out.grad is None: return
        if L.requires_grad:
            g = np.exp(logp)
            g[np.arange(x.shape[0]), lab] -= 1
            g = g * out.grad.reshape(-1, 1) / x.shape[0]
            L.grad = (L.grad if L.grad is not None else 0) + g.reshape(L.shape)
    out._backward = _bw
    return out
