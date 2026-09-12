"""tensorf.losses — like tf.keras.losses (fully differentiable via graph ops)."""
from __future__ import annotations

from .tensor import Tensor, convert_to_tensor, subtract, square, reduce_mean, log
from .ops import _t, clip_by_value
from . import nn as _nn


def mean_squared_error(y_true, y_pred) -> Tensor:
    A, B = _t(y_true), _t(y_pred)
    return reduce_mean(square(subtract(A, B)))


def mean_absolute_error(y_true, y_pred) -> Tensor:
    from .tensor import abs_ as _abs
    A, B = _t(y_true), _t(y_pred)
    return reduce_mean(_abs(subtract(A, B)))


def binary_crossentropy(y_true, y_pred, epsilon=1e-7) -> Tensor:
    from .tensor import add, multiply, negative
    A, B = _t(y_true), _t(y_pred)
    P = clip_by_value(B, epsilon, 1 - epsilon)
    one = 1.0
    term = add(multiply(A, log(P)), multiply(subtract(one, A), log(subtract(one, P))))
    return negative(reduce_mean(term))


def categorical_crossentropy(y_true, y_pred, epsilon=1e-7) -> Tensor:
    from .tensor import multiply, negative, reduce_sum
    A, B = _t(y_true), _t(y_pred)
    P = clip_by_value(B, epsilon, 1 - epsilon)
    return negative(reduce_mean(reduce_sum(multiply(A, log(P)), axis=-1)))


def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, epsilon=1e-7) -> Tensor:
    import numpy as np
    L = _t(y_pred)
    lab = (y_true._data if isinstance(y_true, Tensor) else __import__("numpy").asanyarray(y_true))
    if from_logits:
        per = _nn.sparse_softmax_cross_entropy(L, lab)
    else:
        P = clip_by_value(L, epsilon, 1 - epsilon)
        idx = lab.astype(np.int64).ravel()
        # -mean(log(p_true)) with graph through P (one_hot mask keeps it differentiable)
        from .tensor import one_hot as _oh, reduce_sum as _rs, negative as _neg, reduce_mean as _rm, multiply as _mul, log as _log
        lp = _log(P)
        mask = _oh(idx.reshape(lab.shape), P._data.shape[-1])
        per = _neg(_rs(_mul(mask, lp), axis=-1))
        return _rm(per)
    return reduce_mean(per)


# aliases like TF
mse = mean_squared_error
mae = mean_absolute_error
bce = binary_crossentropy
cce = categorical_crossentropy
sparse_cce = sparse_categorical_crossentropy


def huber(y_true, y_pred, delta=1.0) -> Tensor:
    A, B = _t(y_true), _t(y_pred)
    err = subtract(A, B)
    from .ops import where as _w
    from .tensor import abs_ as _abs
    ae = _abs(err)
    quad = 0.5 * square(err)
    lin = delta * (ae - 0.5 * delta)
    import numpy as _npo
    mask = _npo.asanyarray(ae._data <= delta)
    return reduce_mean(_w(mask, quad, lin))


def hinge(y_true, y_pred) -> Tensor:
    import numpy as _np
    from .ops import maximum as _mx
    from .tensor import convert_to_tensor as _c
    A, B = _t(y_true), _t(y_pred)
    prod = A * B  # y_true in {-1, +1}: max(0, 1 - y*ŷ)
    one = _c(_np.ones_like(prod._data))
    return reduce_mean(_mx(subtract(one, prod), 0.0))
