from __future__ import annotations

import numpy as np


def _smallest_int_dtype(min_val: int, max_val: int) -> np.dtype:
    if min_val >= 0:
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            info = np.iinfo(dtype)
            if max_val <= info.max:
                return dtype
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if min_val >= info.min and max_val <= info.max:
            return dtype
    return np.int64


def downcast_numeric(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "b":
        return arr.astype(np.bool_, copy=False)
    if arr.dtype.kind in ("i", "u"):
        min_val = int(np.min(arr))
        max_val = int(np.max(arr))
        return arr.astype(_smallest_int_dtype(min_val, max_val), copy=False)
    if arr.dtype.kind == "f":
        return arr.astype(np.float32, copy=False)
    raise TypeError(f"Unsupported dtype for downcast: {arr.dtype}")


def ingest_matrix(values: np.ndarray, require_2d: bool = True) -> np.ndarray:
    arr = downcast_numeric(values)
    if require_2d and arr.ndim != 2:
        raise ValueError(f"ingest_matrix expects 2D data, got shape {arr.shape}")
    return np.ascontiguousarray(arr)
