"""tensorf.stats — fast stats for data analysts (numpy only)."""
from __future__ import annotations

import numpy as np


def _np(x):
    return x._data if hasattr(x, "_data") else np.asanyarray(x, dtype=np.float64)


def mean(x, axis=None):
    return float(np.nanmean(_np(x), axis=axis)) if axis is None else np.nanmean(_np(x), axis=axis)


def std(x, axis=None):
    return float(np.nanstd(_np(x), axis=axis)) if axis is None else np.nanstd(_np(x), axis=axis)


def var(x, axis=None):
    return float(np.nanvar(_np(x), axis=axis)) if axis is None else np.nanvar(_np(x), axis=axis)


def median(x, axis=None):
    return np.nanmedian(_np(x), axis=axis)


def percentile(x, q, axis=None):
    return np.nanpercentile(_np(x), q, axis=axis)


def corrcoef(x, y=None):
    a = _np(x).astype(np.float64)
    if y is not None:
        b = _np(y).astype(np.float64)
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return np.corrcoef(a, rowvar=False)


def cov(x, y=None):
    a = _np(x).astype(np.float64)
    if y is not None:
        return float(np.cov(a.ravel(), _np(y).astype(np.float64).ravel())[0, 1])
    return np.cov(a, rowvar=False)


def skew(x):
    a = _np(x).astype(np.float64).ravel()
    m, s = a.mean(), a.std() + 1e-12
    return float(((a - m) ** 3).mean() / s ** 3)


def kurtosis(x):
    a = _np(x).astype(np.float64).ravel()
    m, s = a.mean(), a.std() + 1e-12
    return float(((a - m) ** 4).mean() / s ** 4 - 3)


def zscore(x, axis=0):
    a = _np(x).astype(np.float64)
    return ((a - a.mean(axis, keepdims=True)) / (a.std(axis, keepdims=True) + 1e-12)).astype(np.float32)


def histogram(x, bins=10):
    counts, edges = np.histogram(_np(x), bins=bins)
    return counts, edges


def describe(x) -> dict:
    a = _np(x).astype(np.float64).ravel()
    a = a[~np.isnan(a)]
    return {"count": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std()) if a.size else float("nan"),
            "min": float(a.min()) if a.size else float("nan"),
            "p25": float(np.percentile(a, 25)) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
            "p75": float(np.percentile(a, 75)) if a.size else float("nan"),
            "max": float(a.max()) if a.size else float("nan"),
            "skew": skew(a), "kurtosis": kurtosis(a)}


def correlation_matrix(X):
    return np.corrcoef(_np(X).astype(np.float64), rowvar=False)


__all__ = ["mean", "std", "var", "median", "percentile", "corrcoef", "cov",
           "skew", "kurtosis", "zscore", "histogram", "describe", "correlation_matrix"]
