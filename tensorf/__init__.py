"""tensorf — a faster TensorFlow alternative (CPU) with the same names.

    import tensorf as tf
    x = tf.constant([[1., 2.], [3., 4.]])
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(x * x)
    print(tape.gradient(y, x))

    model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation="relu"),
                                 tf.keras.layers.Dense(10)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=5)

Speed sources (no C/compiler):
- float32 + contiguous by default
- parallel element-wise ops (ThreadPool) for large arrays
- parallel batch-matmul + einsum(optimize=True)
- fused dense/matmul+bias+activation and fused Adam-step
- fully vectorized im2col conv2d
- tf.function with fast-path after tracing + Dataset with prefetch
"""
from __future__ import annotations

import numpy as np

from . import dtypes as dtypes
from .dtypes import (float16, float32, float64, int8, int16, int32, int64,
                     uint8, uint16, uint32, uint64, bool_ as bool, complex64, complex128, string)
from .tensor import (
    Tensor, Variable,
    constant, convert_to_tensor, zeros, ones, zeros_like, ones_like,
    fill, eye, range, linspace, cast, one_hot,
    add, subtract, multiply, divide, pow, negative,
    exp, log, sqrt, square, abs_ as abs,
    matmul, reduce_sum, reduce_mean, reshape, transpose,
)
from .ops import (
    expand_dims, squeeze, concat, concatenate, stack, split, tile, pad,
    gather, where, clip_by_value, maximum, minimum, tensordot, einsum, norm,
    reduce_max, reduce_min, reduce_prod, argmax, argmin, sort, argsort,
    equal, not_equal, greater, greater_equal, less, less_equal,
    linalg,
)
from . import nn as nn
from .nn import (relu, sigmoid, tanh, softplus, leaky_relu, elu, gelu, silu,
                 softmax, log_softmax, dropout, bias_add, dense_fused,
                 conv2d, max_pool, avg_pool, sparse_softmax_cross_entropy)
from .autodiff import GradientTape
from . import optimizers as optimizers
from . import losses as losses
from . import metrics as metrics
from . import layers as layers
from .models import Sequential, Model
from . import keras as keras
from . import data as data
from .data import Dataset
from .function import function, jit
from . import random as random
from . import parallel as parallel
from .parallel import set_workers, get_workers, MAX_WORKERS, PARALLEL_THRESHOLD
# --- data-science / database / analyst stack ---
from . import io as io
from . import database as database
from . import dataframe as dataframe
from .dataframe import DataFrame
from . import preprocessing as preprocessing
from . import stats as stats
from . import ml as ml
from . import text as text
from . import callbacks as callbacks
from . import schedules as schedules
from . import utils as utils
from . import viz as viz
db = database  # short alias: tf.db.Database

abs_ = abs
reduce_sum_fn = reduce_sum

# aliases exactly like TF
concat_alias = concat
maximum_alias = maximum

__version__ = "0.2.0"
__author__ = "salim-slimani"
__copyright__ = "Copyright (c) 2026 salim-slimani"

newaxis = np.newaxis
pi = np.pi
e = np.e
inf = np.inf
nan = np.nan


def shape(t):
    return convert_to_tensor(t).shape


def rank(t):
    return convert_to_tensor(t).ndim


def size(t):
    return convert_to_tensor(t).size


def reshape_alias(a, shape):
    return reshape(a, shape)


def stop_gradient(t):
    from .tensor import Tensor as _T
    a = t._data if isinstance(t, _T) else np.asanyarray(t)
    return _T(a, requires_grad=False)


def zeros_initializer(*a, **k):
    return np.zeros(*a, **k)


def version():
    return __version__


def is_tensor(x):
    return isinstance(x, Tensor)


def convert_dtype(a, dtype):
    return cast(a, dtype)


def info():
    import os
    return {
        "version": __version__,
        "numpy": np.__version__,
        "workers": get_workers(),
        "cpus": os.cpu_count(),
        "default_dtype": "float32",
        "eager": True,
    }


__all__ = [
    "Tensor", "Variable", "constant", "convert_to_tensor", "zeros", "ones",
    "zeros_like", "ones_like", "fill", "eye", "range", "linspace", "cast", "one_hot",
    "add", "subtract", "multiply", "divide", "pow", "negative", "exp", "log",
    "sqrt", "square", "abs", "matmul", "reduce_sum", "reduce_mean", "reshape",
    "transpose", "expand_dims", "squeeze", "concat", "concatenate", "stack",
    "split", "tile", "pad", "gather", "where", "clip_by_value", "maximum",
    "minimum", "tensordot", "einsum", "norm", "reduce_max", "reduce_min",
    "reduce_prod", "argmax", "argmin", "sort", "argsort", "equal", "not_equal",
    "greater", "greater_equal", "less", "less_equal", "linalg", "nn",
    "relu", "sigmoid", "tanh", "softplus", "leaky_relu", "elu", "gelu", "silu",
    "softmax", "log_softmax", "dropout", "bias_add", "dense_fused", "conv2d",
    "max_pool", "avg_pool", "sparse_softmax_cross_entropy",
    "GradientTape", "optimizers", "losses", "metrics", "layers", "Sequential",
    "Model", "keras", "data", "Dataset", "function", "jit", "random",
    "set_workers", "get_workers", "info", "__version__",
    # data-science stack
    "io", "database", "db", "dataframe", "DataFrame", "preprocessing",
    "stats", "ml", "text", "callbacks", "schedules", "utils", "viz",
]
