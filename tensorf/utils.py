"""tensorf.utils — general utilities (seed, to_categorical, save/load...)."""
from __future__ import annotations

import os
import pickle
import random

import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        from .random import set_seed
        set_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def to_categorical(y, num_classes=None):
    from .preprocessing import to_categorical as _tc
    return _tc(y, num_classes)


def to_tensor(x, dtype=np.float32):
    from .tensor import convert_to_tensor
    return convert_to_tensor(x, dtype=dtype)


def to_numpy(x):
    return x._data if hasattr(x, "_data") else np.asanyarray(x)


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_params(model) -> int:
    n = 0
    for v in getattr(model, "trainable_variables", []):
        n += int(np.prod(v.shape))
    return n


def model_summary_string(model) -> str:
    lines = [f"Model: {getattr(model, 'name', '?')}"]
    for l in getattr(model, "layers", []):
        n = sum(int(np.prod(v.shape)) for v in l.trainable_variables)
        lines.append(f" {l.name:25s} {str(getattr(l, 'units', getattr(l, 'filters', ''))):>8s} params: {n}")
    lines.append(f"Total params: {count_params(model)}")
    return "\n".join(lines)


__all__ = ["seed_everything", "to_categorical", "to_tensor", "to_numpy",
           "save", "load", "count_params", "model_summary_string"]
