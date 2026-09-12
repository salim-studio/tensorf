"""tensorf.dataframe — lightweight table for data analysts (pandas-like, no pandas).

    import tensorf as tf
    df = tf.DataFrame({"age": [20, 30, 25], "city": ["oran", "alger", "oran"]})
    df.head()            # print first rows
    df.describe()        # numeric stats
    df["age"].to_numpy()
    df.filter(df["age"] > 21)
    df.groupby("city").mean("age")
    df.to_tensor()       # -> Tensor of numeric cols (for training)
"""
from __future__ import annotations

import csv
from typing import Any, Callable

import numpy as np


def _col_to_np(v):
    from .tensor import Tensor
    if isinstance(v, Tensor):
        return v._data
    return np.asanyarray(list(v) if isinstance(v, (list, tuple)) else v)


class Series:
    def __init__(self, name, values):
        self.name = name
        self.values = _col_to_np(values)

    def to_numpy(self):
        return self.values

    def to_tensor(self):
        from .tensor import convert_to_tensor
        try:
            return convert_to_tensor(self.values.astype(np.float32))
        except Exception:
            return convert_to_tensor(self.values)

    def mean(self):
        return float(np.nanmean(self.values.astype(np.float64)))
    def sum(self):
        return float(np.nansum(self.values.astype(np.float64)))
    def min(self):
        return self.values.min()
    def max(self):
        return self.values.max()
    def std(self):
        return float(np.nanstd(self.values.astype(np.float64)))
    def unique(self):
        return np.unique(self.values)
    def nunique(self):
        return len(np.unique(self.values))
    def isna(self):
        v = self.values
        if v.dtype.kind in "f":
            return np.isnan(v)
        return np.array([x is None for x in v.tolist()])

    def fillna(self, value):
        v = self.values.copy()
        m = self.isna()
        v[m] = value
        return Series(self.name, v)

    # comparisons -> boolean mask (numpy)
    def __gt__(self, o): return self.values > o
    def __ge__(self, o): return self.values >= o
    def __lt__(self, o): return self.values < o
    def __le__(self, o): return self.values <= o
    def __eq__(self, o): return self.values == o  # noqa
    def __ne__(self, o): return self.values != o  # noqa
    def __len__(self): return len(self.values)
    def __repr__(self):
        return f"Series({self.name!r}, {self.values!r})"


class GroupBy:
    def __init__(self, df, key):
        self.df = df
        self.key = key

    def mean(self, col):
        keys = self.df._data[self.key]
        vals = _col_to_np(self.df._data[col]).astype(np.float64)
        uk = np.unique(keys)
        return {k: float(np.nanmean(vals[np.asanyarray(keys) == k])) for k in uk}

    def sum(self, col):
        keys = self.df._data[self.key]
        vals = _col_to_np(self.df._data[col]).astype(np.float64)
        uk = np.unique(keys)
        return {k: float(np.nansum(vals[np.asanyarray(keys) == k])) for k in uk}

    def count(self):
        keys = self.df._data[self.key]
        uk, c = np.unique(keys, return_counts=True)
        return dict(zip(uk.tolist(), c.tolist()))

    def agg(self, col, fn="mean"):
        return getattr(self, fn)(col)


class DataFrame:
    """Column table. Each column is a numpy array (or Tensor on export)."""

    def __init__(self, data=None):
        self._data: dict[str, np.ndarray] = {}
        if data is None:
            return
        if isinstance(data, dict):
            for k, v in data.items():
                self._data[str(k)] = _col_to_np(v)
        else:
            raise TypeError("DataFrame(data: dict)")

    # -- constructors --
    @staticmethod
    def from_dict(d) -> "DataFrame":
        return DataFrame(d)

    @staticmethod
    def from_tensors(d) -> "DataFrame":
        from .tensor import Tensor
        out = DataFrame()
        for k, v in d.items():
            out._data[str(k)] = v._data if isinstance(v, Tensor) else np.asanyarray(v)
        return out

    @staticmethod
    def read_csv(path, delimiter=",", header=True, encoding="utf-8") -> "DataFrame":
        with open(path, newline="", encoding=encoding) as f:
            rdr = csv.reader(f, delimiter=delimiter)
            rows = list(rdr)
        if not rows:
            return DataFrame({})
        if header:
            cols, body = rows[0], rows[1:]
        else:
            cols, body = [f"c{i}" for i in range(len(rows[0]))], rows
        data = {c: [] for c in cols}
        for r in body:
            for c, v in zip(cols, r):
                try:
                    data[c].append(float(v))
                except Exception:
                    data[c].append(v)
        # numeric-ify columns when possible
        for c in cols:
            try:
                data[c] = np.asarray(data[c], dtype=np.float64)
                if np.isnan(data[c]).any():
                    raise ValueError
            except Exception:
                # keep mixed; try int/float else object
                try:
                    data[c] = np.asarray(data[c])
                except Exception:
                    pass
        return DataFrame(data)

    @staticmethod
    def from_pandas(pdf) -> "DataFrame":
        out = DataFrame()
        for c in pdf.columns:
            out._data[str(c)] = pdf[c].to_numpy()
        return out

    # -- basics --
    @property
    def columns(self) -> list:
        return list(self._data.keys())

    @property
    def shape(self):
        n = len(next(iter(self._data.values()))) if self._data else 0
        return (n, len(self._data))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(key, self._data[key])
        if isinstance(key, (list, tuple)):
            return DataFrame({k: self._data[k] for k in key})
        # boolean mask
        m = np.asanyarray(key)
        if m.dtype == bool:
            return self._mask(m)
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._data[str(key)] = _col_to_np(value)

    def _mask(self, m) -> "DataFrame":
        return DataFrame({k: np.asanyarray(v)[m] for k, v in self._data.items()})

    def filter(self, mask) -> "DataFrame":
        m = np.asanyarray(mask.values if isinstance(mask, Series) else mask, dtype=bool)
        return self._mask(m)

    def head(self, n=5):
        sub = {k: v[:n] for k, v in self._data.items()}
        print(self._fmt(sub))
        return DataFrame(sub)

    def tail(self, n=5):
        sub = {k: v[-n:] for k, v in self._data.items()}
        print(self._fmt(sub))
        return DataFrame(sub)

    def _fmt(self, sub=None) -> str:
        d = sub if sub is not None else self._data
        cols = list(d.keys())
        lines = [" | ".join(cols)]
        n = len(next(iter(d.values()))) if d else 0
        for i in range(min(n, 20)):
            lines.append(" | ".join(str(d[c][i]) for c in cols))
        if n > 20:
            lines.append(f"... ({n} rows)")
        return "\n".join(lines)

    def __repr__(self):
        return f"DataFrame(shape={self.shape}, columns={self.columns})\n" + self._fmt()

    # -- analysis --
    def describe(self) -> "DataFrame":
        stats = {"stat": ["count", "mean", "std", "min", "max"]}
        for c, v in self._data.items():
            if np.asanyarray(v).dtype.kind in "iuf":
                a = np.asanyarray(v, dtype=np.float64)
                stats[c] = [float(np.sum(~np.isnan(a))), float(np.nanmean(a)),
                            float(np.nanstd(a)), float(np.nanmin(a)), float(np.nanmax(a))]
        return DataFrame(stats)

    def dtypes(self) -> dict:
        return {k: str(np.asanyarray(v).dtype) for k, v in self._data.items()}

    def isna(self) -> dict:
        out = {}
        for k, v in self._data.items():
            a = np.asanyarray(v)
            out[k] = int(np.isnan(a.astype(np.float64)).sum()) if a.dtype.kind in "iuf" else int(sum(x is None for x in a.tolist()))
        return out

    def dropna(self) -> "DataFrame":
        n = len(self)
        keep = np.ones(n, dtype=bool)
        for v in self._data.values():
            a = np.asanyarray(v)
            if a.dtype.kind in "f":
                keep &= ~np.isnan(a.astype(np.float64))
            elif a.dtype.kind in "iu":
                pass
            else:
                keep &= np.array([x is not None for x in a.tolist()])
        return self._mask(keep)

    def fillna(self, value=0.0) -> "DataFrame":
        out = DataFrame()
        for k, v in self._data.items():
            a = np.asanyarray(v).copy()
            if a.dtype.kind == "f":
                a[np.isnan(a)] = value
            out._data[k] = a
        return out

    def sort_values(self, by, ascending=True) -> "DataFrame":
        idx = np.argsort(np.asanyarray(self._data[by]), kind="stable")
        if not ascending:
            idx = idx[::-1]
        return DataFrame({k: np.asanyarray(v)[idx] for k, v in self._data.items()})

    def groupby(self, key) -> GroupBy:
        return GroupBy(self, key)

    def value_counts(self, col) -> dict:
        uk, c = np.unique(np.asanyarray(self._data[col]), return_counts=True)
        return dict(zip(uk.tolist(), c.tolist()))

    def corr(self) -> dict:
        num = [k for k, v in self._data.items() if np.asanyarray(v).dtype.kind in "iuf"]
        if len(num) < 2:
            return {}
        m = np.stack([np.asanyarray(self._data[k], dtype=np.float64) for k in num])
        return {k: row for k, row in zip(num, np.corrcoef(m).tolist())}

    def merge(self, other, on, how="inner") -> "DataFrame":
        a_key = np.asanyarray(self._data[on])
        b_key = np.asanyarray(other._data[on])
        b_idx = {v: i for i, v in enumerate(b_key.tolist())}
        out = {k: [] for k in list(self._data.keys()) + [k for k in other._data.keys() if k != on]}
        for i, v in enumerate(a_key.tolist()):
            j = b_idx.get(v)
            if j is None and how == "inner":
                continue
            for k in self._data:
                out[k].append(self._data[k][i])
            for k in other._data:
                if k == on:
                    continue
                out[k].append(other._data[k][j] if j is not None else None)
        return DataFrame({k: np.asanyarray(v) for k, v in out.items()})

    # -- ML export --
    def to_numpy(self, columns=None) -> np.ndarray:
        cols = columns or [k for k, v in self._data.items() if np.asanyarray(v).dtype.kind in "iuf"]
        return np.stack([np.asanyarray(self._data[k], dtype=np.float32) for k in cols], axis=1) if cols else np.empty((len(self), 0), np.float32)

    def to_tensor(self, columns=None):
        from .tensor import convert_to_tensor
        return convert_to_tensor(self.to_numpy(columns))

    def _matrix(self, dtype=float):
        cols = self.columns
        return cols, [[row[i] if hasattr(row, "__len__") else row for i in range(len(cols))] for row in zip(*[np.asanyarray(self._data[c]).tolist() for c in cols])]

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("to_pandas needs pandas: pip install 'tensorf[io]'") from e
        return pd.DataFrame({k: np.asanyarray(v) for k, v in self._data.items()})

    def to_csv(self, path, **kw) -> str:
        import csv as _csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(self.columns)
            for i in range(len(self)):
                w.writerow([np.asanyarray(self._data[c])[i] for c in self.columns])
        return path

    def to_sql(self, table, con, if_exists="replace"):
        from .database import to_sql as _ts
        return _ts(table, self, con, if_exists)


__all__ = ["DataFrame", "Series", "GroupBy"]
