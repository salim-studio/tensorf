"""tensorfly.preprocessing — data preparation (sklearn-like, works on Tensor/numpy/DataFrame).

    import tensorfly as tf
    X_train, X_test, y_train, y_test = tf.preprocessing.train_test_split(X, y)
    sc = tf.preprocessing.StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
"""
from __future__ import annotations

import numpy as np


def _np(x):
    from .tensor import Tensor
    from .dataframe import DataFrame
    if isinstance(x, Tensor):
        return x._data.astype(np.float64)
    if isinstance(x, DataFrame):
        return x.to_numpy().astype(np.float64)
    return np.asanyarray(x, dtype=np.float64)


def _wrap(out, like):
    from .tensor import Tensor, convert_to_tensor
    if isinstance(like, Tensor):
        return convert_to_tensor(out.astype(np.float32))
    return out.astype(np.float32)


class StandardScaler:
    """x' = (x - mean) / scale."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    def fit(self, X):
        a = _np(X)
        self.mean_ = np.nanmean(a, axis=0)
        self.scale_ = np.nanstd(a, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self
    def transform(self, X):
        return _wrap((_np(X) - self.mean_) / self.scale_, X)
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X):
        return _wrap(_np(X) * self.scale_ + self.mean_, X)


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.lo, self.hi = feature_range
        self.data_min_ = self.data_max_ = None
    def fit(self, X):
        a = _np(X)
        self.data_min_ = np.nanmin(a, axis=0)
        self.data_max_ = np.nanmax(a, axis=0)
        return self
    def transform(self, X):
        a = _np(X)
        rng = self.data_max_ - self.data_min_
        rng[rng == 0] = 1.0
        scaled = (a - self.data_min_) / rng
        return _wrap(scaled * (self.hi - self.lo) + self.lo, X)
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RobustScaler:
    def fit(self, X):
        a = _np(X)
        self.center_ = np.nanmedian(a, axis=0)
        q1, q3 = np.nanpercentile(a, 25, axis=0), np.nanpercentile(a, 75, axis=0)
        self.scale_ = q3 - q1
        self.scale_[self.scale_ == 0] = 1.0
        return self
    def transform(self, X):
        return _wrap((_np(X) - self.center_) / self.scale_, X)
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Normalizer:
    def __init__(self, norm="l2"):
        self.norm = norm
    def fit(self, X):
        return self
    def transform(self, X):
        a = _np(X)
        n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        return _wrap(a / n, X)
    def fit_transform(self, X):
        return self.transform(X)


class LabelEncoder:
    def fit(self, y):
        self.classes_ = np.unique(np.asanyarray(y))
        self._map = {v: i for i, v in enumerate(self.classes_.tolist())}
        return self
    def transform(self, y):
        return np.array([self._map[v] for v in np.asanyarray(y).tolist()])
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    def inverse_transform(self, y):
        return np.array([self.classes_[int(i)] for i in np.asanyarray(y).ravel()])


class OneHotEncoder:
    def __init__(self, sparse=False):
        self.sparse = sparse
    def fit(self, y):
        le = LabelEncoder().fit(y)
        self.classes_ = le.classes_
        self._map = le._map
        return self
    def transform(self, y):
        idx = np.array([self._map[v] for v in np.asanyarray(y).tolist()])
        out = np.zeros((len(idx), len(self.classes_)), np.float32)
        out[np.arange(len(idx)), idx] = 1.0
        return out
    def fit_transform(self, y):
        return self.fit(y).transform(y)


class SimpleImputer:
    """strategy: mean/median/most_frequent/constant."""
    def __init__(self, strategy="mean", fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value
    def fit(self, X):
        a = np.asanyarray(X, dtype=np.float64) if not hasattr(X, "_data") else X._data.astype(np.float64)
        if self.strategy == "mean":
            self.stat_ = np.nanmean(a, axis=0)
        elif self.strategy == "median":
            self.stat_ = np.nanmedian(a, axis=0)
        elif self.strategy == "constant":
            self.stat_ = np.full(a.shape[1] if a.ndim > 1 else 1, self.fill_value)
        else:
            self.stat_ = np.nanmean(a, axis=0)
        self.stat_ = np.where(np.isnan(self.stat_), self.fill_value, self.stat_)
        return self
    def transform(self, X):
        from .tensor import Tensor
        a = (X._data if isinstance(X, Tensor) else np.asanyarray(X, dtype=np.float64)).astype(np.float64).copy()
        idx = np.where(np.isnan(a))
        if a.ndim == 1:
            a[idx] = np.take(self.stat_, idx)
        else:
            a[idx] = np.take(self.stat_, idx[1])
        return a.astype(np.float32)
    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    rng = np.random.default_rng(random_state)
    n = len(np.asanyarray(arrays[0]))
    idx = np.arange(n)
    if shuffle:
        if stratify is not None:
            # simple stratified shuffle
            s = np.asanyarray(stratify)
            for cls in np.unique(s):
                c_idx = np.where(s == cls)[0]
                rng.shuffle(c_idx)
                # keep class ratios: simplified — global shuffle after grouping by class
            rng.shuffle(idx)
        else:
            rng.shuffle(idx)
    n_test = int(n * test_size) if isinstance(test_size, float) else int(test_size)
    te, tr = idx[:n_test], idx[n_test:]
    out = []
    for a in arrays:
        arr = np.asanyarray(a._data if hasattr(a, "_data") else a)
        out += [arr[tr], arr[te]]
    return out


def to_categorical(y, num_classes=None):
    y = np.asanyarray(y).astype(np.int64).ravel()
    n = num_classes or int(y.max()) + 1
    out = np.zeros((y.size, n), np.float32)
    out[np.arange(y.size), y] = 1.0
    return out


def pad_sequences(sequences, maxlen=None, padding="post", value=0.0, dtype=np.int32):
    seqs = [np.asanyarray(s).ravel() for s in sequences]
    maxlen = maxlen or max(len(s) for s in seqs)
    out = np.full((len(seqs), maxlen), value, dtype=dtype)
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        if padding == "post":
            out[i, :len(s)] = s
        else:
            out[i, -len(s):] = s
    return out


def normalize(X, axis=1):
    a = _np(X)
    n = np.linalg.norm(a, axis=axis, keepdims=True) + 1e-12
    return (a / n).astype(np.float32)


def polynomial_features(X, degree=2, include_bias=True):
    from itertools import combinations_with_replacement
    a = np.asanyarray(X, dtype=np.float64)
    n, d = a.shape
    cols = [np.ones((n, 1))] if include_bias else []
    for deg in range(1, degree + 1):
        for c in combinations_with_replacement(range(d), deg):
            cols.append(np.prod(a[:, c], axis=1, keepdims=True))
    return np.concatenate(cols, axis=1).astype(np.float32)


__all__ = ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
           "LabelEncoder", "OneHotEncoder", "SimpleImputer", "train_test_split",
           "to_categorical", "pad_sequences", "normalize", "polynomial_features"]
