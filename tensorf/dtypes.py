"""tensorf.dtypes — type names like tf.*"""
from __future__ import annotations

import numpy as np

float16 = np.float16
float32 = np.float32
float64 = np.float64
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64
bool_ = np.bool_
bool = np.bool_  # noqa: A001  (tf.bool)
complex64 = np.complex64
complex128 = np.complex128
string = np.str_

_DTYPE_MAP = {
    "float16": np.float16, "float32": np.float32, "float64": np.float64,
    "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "bool": np.bool_, "complex64": np.complex64,
}


def as_dtype(d):
    if d is None:
        return None
    if isinstance(d, np.dtype):
        return d
    if isinstance(d, type) and issubclass(d, np.generic):
        return np.dtype(d)
    if isinstance(d, str):
        return np.dtype(_DTYPE_MAP.get(d, d))
    try:
        return np.dtype(d)
    except Exception:
        return None
