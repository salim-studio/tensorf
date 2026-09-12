"""tensorf.io — data loading/saving for developers and analysts (like pandas/numpy I/O).

    import tensorf as tf
    X = tf.io.load_csv("data.csv", header=True)        # -> Tensor
    tf.io.save_csv("out.csv", X)
    d = tf.io.load_json("config.json")
    arr = tf.io.load_npy("x.npy")

Depends only on numpy + stdlib. pandas/pyarrow are optional (fast parquet/csv).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np


def _to_tensor(a, dtype=np.float32):
    from .tensor import Tensor
    return Tensor(np.asanyarray(a), dtype=None) if isinstance(dtype, type(None)) else __import__("tensorf.tensor", fromlist=["convert_to_tensor"]).convert_to_tensor(a, dtype=dtype)


def load_csv(path, delimiter=",", header=False, dtype=np.float32, usecols=None,
             skiprows=0, encoding="utf-8") -> "Tensor":
    """Load numeric CSV -> Tensor(float32). Non-numeric cells become NaN (use DataFrame for mixed types)."""
    from .tensor import convert_to_tensor
    rows = []
    with open(path, newline="", encoding=encoding) as f:
        rdr = csv.reader(f, delimiter=delimiter)
        for _ in range(skiprows):
            next(rdr, None)
        if header:
            next(rdr, None)
        for row in rdr:
            if usecols is not None:
                row = [row[i] for i in usecols]
            vals = []
            for c in row:
                try:
                    vals.append(float(c))
                except Exception:
                    vals.append(np.nan)
            rows.append(vals)
    a = np.asarray(rows, dtype=np.float32 if dtype is None else dtype)
    return convert_to_tensor(a, dtype=dtype)


def save_csv(path, data, delimiter=",", header=None, fmt="%.6g") -> str:
    from .tensor import Tensor
    a = data._data if isinstance(data, Tensor) else np.asanyarray(data)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delimiter)
        if header:
            w.writerow(list(header) if not isinstance(header, str) else [header])
        if a.ndim == 1:
            for v in a:
                w.writerow([f"{v:{fmt[1:]}}" if isinstance(fmt, str) and fmt.startswith("%") else v])
        else:
            for row in a:
                w.writerow([f"{v:{fmt[1:]}c}".replace("c", "") if False else (fmt % v if isinstance(fmt, str) and "%" in fmt else v) for v in row])
    return path


def load_txt(path, delimiter=None, dtype=np.float32, skiprows=0) -> "Tensor":
    from .tensor import convert_to_tensor
    a = np.loadtxt(path, delimiter=delimiter, dtype=np.float32, skiprows=skiprows)
    return convert_to_tensor(a, dtype=dtype)


def save_txt(path, data, fmt="%.6g", delimiter=" ") -> str:
    from .tensor import Tensor
    a = data._data if isinstance(data, Tensor) else np.asanyarray(data)
    np.savetxt(path, a, fmt=fmt, delimiter=delimiter)
    return path


def load_npy(path) -> "Tensor":
    from .tensor import convert_to_tensor
    return convert_to_tensor(np.load(path, allow_pickle=True))


def save_npy(path, data) -> str:
    from .tensor import Tensor
    a = data._data if isinstance(data, Tensor) else np.asanyarray(data)
    np.save(path, a)
    return path


def load_npz(path) -> dict:
    from .tensor import convert_to_tensor
    z = np.load(path, allow_pickle=True)
    return {k: convert_to_tensor(z[k]) for k in z.files}


def save_npz(path, **arrays) -> str:
    from .tensor import Tensor
    np.savez(path, **{k: (v._data if isinstance(v, Tensor) else np.asanyarray(v)) for k, v in arrays.items()})
    return path


def load_json(path, encoding="utf-8") -> Any:
    with open(path, encoding=encoding) as f:
        return json.load(f)


def save_json(path, obj, indent=2, encoding="utf-8") -> str:
    def _default(o):
        try:
            from .tensor import Tensor as _T
            if isinstance(o, _T):
                return o._data.tolist()
        except Exception:
            pass
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not JSON serializable: {type(o)}")
    with open(path, "w", encoding=encoding) as f:
        json.dump(obj, f, indent=indent, default=_default)
    return path


def read_parquet(path, columns=None):
    """Read parquet -> DataFrame (needs pandas+pyarrow, else a clear ImportError)."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("tf.io.read_parquet needs pandas+pyarrow: pip install 'tensorf[io]'") from e
    df = pd.read_parquet(path, columns=columns)
    from .dataframe import DataFrame
    return DataFrame.from_pandas(df)


def write_parquet(path, data, **kw) -> str:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("tf.io.write_parquet needs pandas+pyarrow: pip install 'tensorf[io]'") from e
    from .tensor import Tensor
    from .dataframe import DataFrame
    if isinstance(data, DataFrame):
        data.to_pandas().to_parquet(path, **kw)
    elif isinstance(data, Tensor):
        pd.DataFrame(data._data).to_parquet(path, **kw)
    else:
        pd.DataFrame(np.asanyarray(data)).to_parquet(path, **kw)
    return path


def load_image(path, dtype=np.float32):
    """Load image via PIL (optional) -> Tensor HWC float32 in [0,1]."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("tf.io.load_image needs pillow: pip install pillow") from e
    from .tensor import convert_to_tensor
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return convert_to_tensor(im.astype(dtype or np.float32))


__all__ = ["load_csv", "save_csv", "load_txt", "save_txt", "load_npy", "save_npy",
           "load_npz", "save_npz", "load_json", "save_json", "read_parquet",
           "write_parquet", "load_image"]
