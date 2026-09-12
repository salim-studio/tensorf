"""tensorf.layers — like tf.keras.layers."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor, Variable, convert_to_tensor
from . import nn as _nn
from .ops import _t


class Layer:
    def __init__(self, name=""):
        self.name = name or self.__class__.__name__
        self._built = False
        self.trainable_variables: list[Variable] = []
        self._training = True

    def build(self, input_shape):
        self._built = True

    def __call__(self, x, training=None):
        if training is None:
            training = self._training
        t = x if isinstance(x, Tensor) else convert_to_tensor(x)
        if not self._built:
            self.build(t.shape)
        return self.call(t, training=training)

    def call(self, x, training=True):
        raise NotImplementedError

    def get_weights(self):
        return [v.numpy().copy() for v in self.trainable_variables]

    def set_weights(self, weights):
        for v, w in zip(self.trainable_variables, weights):
            v.assign(w)

    @property
    def trainable_weights(self):
        return self.trainable_variables


def _get_act(name):
    if name is None or name == "linear":
        return None
    m = {"relu": _nn.relu, "sigmoid": _nn.sigmoid, "tanh": _nn.tanh,
         "softmax": _nn.softmax, "gelu": _nn.gelu, "silu": _nn.silu,
         "softplus": _nn.softplus, "leaky_relu": _nn.leaky_relu, "elu": _nn.elu}
    if callable(name):
        return name
    if name not in m:
        raise ValueError(f"unknown activation '{name}'")
    return m[name]


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros", name=""):
        super().__init__(name)
        self.units = units
        self.activation = activation
        self._act_fn = _get_act(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        fan_in = int(input_shape[-1])
        rng = np.random.default_rng()
        if self.kernel_initializer == "zeros":
            W = np.zeros((fan_in, self.units), np.float32)
        elif self.kernel_initializer == "ones":
            W = np.ones((fan_in, self.units), np.float32)
        else:  # glorot_uniform
            lim = np.sqrt(6 / (fan_in + self.units))
            W = rng.uniform(-lim, lim, (fan_in, self.units)).astype(np.float32)
        self.kernel = Variable(W, name="kernel")
        self.trainable_variables.append(self.kernel)
        if self.use_bias:
            self.bias = Variable(np.zeros(self.units, np.float32), name="bias")
            self.trainable_variables.append(self.bias)
        else:
            self.bias = None
        self._built = True

    def call(self, x, training=True):
        y = _nn.dense_fused(x, self.kernel, self.bias,
                            activation=self.activation if isinstance(self.activation, str) else None)
        if self.activation is not None and not isinstance(self.activation, str):
            y = self._act_fn(y)
        return y


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="VALID", activation=None, use_bias=True, name=""):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self._act_fn = _get_act(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        Cin = int(input_shape[-1])
        kh, kw = self.kernel_size
        lim = np.sqrt(6 / (kh * kw * Cin + self.filters))
        W = np.random.default_rng().uniform(-lim, lim, (kh, kw, Cin, self.filters)).astype(np.float32)
        self.kernel = Variable(W)
        self.trainable_variables.append(self.kernel)
        if self.use_bias:
            self.bias = Variable(np.zeros(self.filters, np.float32))
            self.trainable_variables.append(self.bias)
        else:
            self.bias = None
        self._built = True

    def call(self, x, training=True):
        y = _nn.conv2d(x, self.kernel, self.strides, self.padding)
        if self.use_bias:
            b = self.bias._data.reshape(1, 1, 1, -1)
            from .tensor import add
            y = add(y, b)
        if self._act_fn is not None:
            y = self._act_fn(y)
        return y


class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, strides=None, padding="VALID", name=""):
        super().__init__(name)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def call(self, x, training=True):
        return _nn.max_pool(x, self.pool_size, self.strides, self.padding)


class AveragePooling2D(Layer):
    def __init__(self, pool_size=2, strides=None, padding="VALID", name=""):
        super().__init__(name)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def call(self, x, training=True):
        return _nn.avg_pool(x, self.pool_size, self.strides, self.padding)


class Flatten(Layer):
    def call(self, x, training=True):
        from .tensor import reshape
        X = _t(x)
        return reshape(X, (X.shape[0], -1))


class Dropout(Layer):
    def __init__(self, rate=0.5, seed=None, name=""):
        super().__init__(name)
        self.rate = rate
        self.seed = seed

    def call(self, x, training=True):
        return _nn.dropout(x, self.rate, training=training, seed=self.seed)


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, name=""):
        super().__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        W = np.random.default_rng().standard_normal((self.input_dim, self.output_dim)).astype(np.float32) * 0.05
        self.embeddings = Variable(W)
        self.trainable_variables.append(self.embeddings)
        self._built = True

    def call(self, x, training=True):
        from .ops import gather
        return gather(self.embeddings, np.asanyarray(x._data if isinstance(x, Tensor) else x))


class BatchNormalization(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-3, name=""):
        super().__init__(name)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.gamma = Variable(np.ones(d, np.float32))
        self.beta = Variable(np.zeros(d, np.float32))
        self.trainable_variables += [self.gamma, self.beta]
        self.moving_mean = np.zeros(d, np.float32)
        self.moving_var = np.ones(d, np.float32)
        self._built = True

    def call(self, x, training=True):
        X = _t(x)
        if training:
            mu = X._data.mean(axis=0)
            va = X._data.var(axis=0)
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mu
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * va
            xn = (X._data - mu) / np.sqrt(va + self.epsilon)
        else:
            xn = (X._data - self.moving_mean) / np.sqrt(self.moving_var + self.epsilon)
        from .tensor import convert_to_tensor as _c
        return _c(xn * self.gamma._data + self.beta._data)


class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-3, name=""):
        super().__init__(name)
        self.epsilon = epsilon

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.gamma = Variable(np.ones(d, np.float32))
        self.beta = Variable(np.zeros(d, np.float32))
        self.trainable_variables += [self.gamma, self.beta]
        self._built = True

    def call(self, x, training=True):
        X = _t(x)
        mu = X._data.mean(axis=-1, keepdims=True)
        va = X._data.var(axis=-1, keepdims=True)
        from .tensor import convert_to_tensor as _c
        return _c((X._data - mu) / np.sqrt(va + self.epsilon) * self.gamma._data + self.beta._data)


class SimpleRNN(Layer):
    def __init__(self, units, activation="tanh", name=""):
        super().__init__(name)
        self.units = units
        self._act_fn = _get_act(activation)

    def build(self, input_shape):
        f = int(input_shape[-1])
        rng = np.random.default_rng()
        self.Wx = Variable((rng.standard_normal((f, self.units)) * np.sqrt(1 / f)).astype(np.float32))
        self.Wh = Variable((rng.standard_normal((self.units, self.units)) * np.sqrt(1 / self.units)).astype(np.float32))
        self.b = Variable(np.zeros(self.units, np.float32))
        self.trainable_variables += [self.Wx, self.Wh, self.b]
        self._built = True

    def call(self, x, training=True):
        X = _t(x)  # (B, T, F)
        from .tensor import matmul, add
        h = None
        for t in range(X.shape[1]):
            xt = X[:, t, :]
            pre = add(matmul(xt, self.Wx), self.b)
            if h is not None:
                pre = add(pre, matmul(h, self.Wh))
            h = self._act_fn(pre)
        return h


class InputLayer(Layer):
    def __init__(self, shape, name=""):
        super().__init__(name or "input")
        self.input_shape = shape
        self._built = True

    def call(self, x, training=True):
        return _t(x)


def Input(shape, name="input"):
    layer = InputLayer(shape, name)
    layer._keras_input = True
    layer._shape = shape
    return layer


class Activation(Layer):
    def __init__(self, activation, name=""):
        super().__init__(name)
        self._act_fn = _get_act(activation)
        self._built = True

    def call(self, x, training=True):
        return self._act_fn(_t(x))


class Reshape(Layer):
    def __init__(self, target_shape, name=""):
        super().__init__(name or "reshape")
        self.target_shape = tuple(target_shape)
        self._built = True

    def call(self, x, training=True):
        from .tensor import reshape as _rs
        X = _t(x)
        return _rs(X, (X.shape[0],) + self.target_shape)


class GlobalAveragePooling2D(Layer):
    def __init__(self, name=""):
        super().__init__(name or "gap")
        self._built = True

    def call(self, x, training=True):
        from .tensor import reduce_mean
        X = _t(x)
        return reduce_mean(X, axis=(1, 2))


class GlobalMaxPooling2D(Layer):
    def __init__(self, name=""):
        super().__init__(name or "gmp")
        self._built = True

    def call(self, x, training=True):
        import numpy as _np
        from .tensor import convert_to_tensor as _c
        X = _t(x)
        return _c(_np.asarray(X._data).max(axis=(1, 2)))


class Add(Layer):
    def __init__(self, name=""):
        super().__init__(name or "add")
        self._built = True

    def call(self, x, training=True):
        from .tensor import add as _add
        if isinstance(x, (list, tuple)):
            out = _t(x[0])
            for v in x[1:]:
                out = _add(out, _t(v))
            return out
        return _t(x)


class Concatenate(Layer):
    def __init__(self, axis=-1, name=""):
        super().__init__(name or "concat")
        self.axis = axis
        self._built = True

    def call(self, x, training=True):
        from .ops import concat as _cc
        return _cc(list(x), axis=self.axis)


class AveragePooling2D(MaxPooling2D):
    pass


class LSTM(Layer):
    """Lightweight LSTM (B,T,F) -> (B,units)."""

    def __init__(self, units, return_sequences=False, name=""):
        super().__init__(name)
        self.units = units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        f = int(input_shape[-1])
        rng = np.random.default_rng()
        s = np.sqrt(1 / max(1, f + self.units))
        self.W = Variable((rng.standard_normal((f, 4 * self.units)) * s).astype(np.float32))
        self.U = Variable((rng.standard_normal((self.units, 4 * self.units)) * s).astype(np.float32))
        self.b = Variable(np.zeros(4 * self.units, np.float32))
        self.trainable_variables += [self.W, self.U, self.b]
        self._built = True

    def call(self, x, training=True):
        from .tensor import matmul, add
        from . import nn as _nmod
        _sig, _tanh = _nmod.sigmoid, _nmod.tanh
        X = _t(x)
        B, T, _ = X.shape
        h = X[:, 0:1, :] * 0  # placeholder graph-safe zeros
        import numpy as _np
        # numpy loop rebuilt as a differentiable op graph (each step differentiable)
        h_t = None
        c_t = None
        seq = []
        Wd = self.W._data
        Ud = self.U._data
        bd = self.b._data
        # unroll each timestep with numpy, then rebuild the graph via matmul/add/sigmoid/tanh
        for t in range(T):
            xt = X[:, t, :]
            gates = add(add(matmul(xt, self.W), matmul(h_t, self.U) if h_t is not None else 0.0), self.b)
            # split gates (slice hält graph via _slice)
            i = _sig(gates[:, 0:self.units])
            f = _sig(gates[:, self.units:2 * self.units])
            o = _sig(gates[:, 2 * self.units:3 * self.units])
            g = _tanh(gates[:, 3 * self.units:4 * self.units])
            c_t = g if c_t is None else add(f * c_t, i * g)
            h_t = o * _tanh(c_t)
            seq.append(h_t)
        if self.return_sequences:
            from .ops import stack as _st
            from .tensor import transpose as _tp, reshape as _rs
            s = _st(seq, axis=0)  # (T,B,U)
            return _tp(s, (1, 0, 2))
        return h_t


class GRU(Layer):
    """Lightweight GRU (B,T,F) -> (B,units)."""

    def __init__(self, units, return_sequences=False, name=""):
        super().__init__(name)
        self.units = units
        self.return_sequences = return_sequences

    def build(self, input_shape):
        f = int(input_shape[-1])
        rng = np.random.default_rng()
        s = np.sqrt(1 / max(1, f))
        self.Wz = Variable((rng.standard_normal((f, self.units)) * s).astype(np.float32))
        self.Wr = Variable((rng.standard_normal((f, self.units)) * s).astype(np.float32))
        self.Wh = Variable((rng.standard_normal((f, self.units)) * s).astype(np.float32))
        self.Uz = Variable((rng.standard_normal((self.units, self.units)) * s).astype(np.float32))
        self.Ur = Variable((rng.standard_normal((self.units, self.units)) * s).astype(np.float32))
        self.Uh = Variable((rng.standard_normal((self.units, self.units)) * s).astype(np.float32))
        self.bz = Variable(np.zeros(self.units, np.float32))
        self.br = Variable(np.zeros(self.units, np.float32))
        self.bh = Variable(np.zeros(self.units, np.float32))
        self.trainable_variables += [self.Wz, self.Wr, self.Wh, self.Uz, self.Ur, self.Uh,
                                     self.bz, self.br, self.bh]
        self._built = True

    def call(self, x, training=True):
        from .tensor import matmul, add
        from . import nn as _nmod
        _sig, _tanh = _nmod.sigmoid, _nmod.tanh
        X = _t(x)
        T = X.shape[1]
        h_t = None
        seq = []
        for t in range(T):
            xt = X[:, t, :]
            z = _sig(add(add(matmul(xt, self.Wz), matmul(h_t, self.Uz) if h_t is not None else 0.0), self.bz))
            r = _sig(add(add(matmul(xt, self.Wr), matmul(h_t, self.Ur) if h_t is not None else 0.0), self.br))
            rec = matmul(r * h_t, self.Uh) if h_t is not None else 0.0
            hh = _tanh(add(add(matmul(xt, self.Wh), rec), self.bh))
            h_t = hh if h_t is None else add((1.0 - z) * h_t, z * hh)
            seq.append(h_t)
        if self.return_sequences:
            from .ops import stack as _st
            from .tensor import transpose as _tp
            return _tp(_st(seq, axis=0), (1, 0, 2))
        return h_t


class MultiHeadAttention(Layer):
    """Multi-head self-attention (B,T,D)."""

    def __init__(self, num_heads=4, key_dim=32, name=""):
        super().__init__(name)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        d = int(input_shape[-1])
        rng = np.random.default_rng()
        s = np.sqrt(1 / max(1, d))
        self.Wq = Variable((rng.standard_normal((d, self.num_heads * self.key_dim)) * s).astype(np.float32))
        self.Wk = Variable((rng.standard_normal((d, self.num_heads * self.key_dim)) * s).astype(np.float32))
        self.Wv = Variable((rng.standard_normal((d, self.num_heads * self.key_dim)) * s).astype(np.float32))
        self.Wo = Variable((rng.standard_normal((self.num_heads * self.key_dim, d)) * s).astype(np.float32))
        self.trainable_variables += [self.Wq, self.Wk, self.Wv, self.Wo]
        self._built = True

    def call(self, x, training=True):
        from .tensor import matmul, reshape, transpose
        from . import nn as _nn
        X = _t(x)
        B, T, _ = X.shape
        Q = reshape(matmul(X, self.Wq), (B, T, self.num_heads, self.key_dim))
        K = reshape(matmul(X, self.Wk), (B, T, self.num_heads, self.key_dim))
        V = reshape(matmul(X, self.Wv), (B, T, self.num_heads, self.key_dim))
        # scores per head via numpy-graph: (B,H,T,T)
        import numpy as _np
        Qd = Q._data.transpose(0, 2, 1, 3).reshape(B * self.num_heads, T, self.key_dim)
        Kd = K._data.transpose(0, 2, 1, 3).reshape(B * self.num_heads, T, self.key_dim)
        Vd = V._data.transpose(0, 2, 1, 3).reshape(B * self.num_heads, T, self.key_dim)
        from .tensor import convert_to_tensor as _c
        scores = _c(Qd @ Kd.transpose(0, 2, 1) / _np.sqrt(self.key_dim))
        attn = _nn.softmax(scores, axis=-1)
        ctx = matmul(attn, _c(Vd))  # (B*H,T,K)
        ctx = _c(ctx._data.reshape(B, self.num_heads, T, self.key_dim).transpose(0, 2, 1, 3).reshape(B, T, -1))
        return matmul(ctx, self.Wo)
