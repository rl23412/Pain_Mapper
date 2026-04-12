from __future__ import annotations

import math
import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from scipy import signal
from scipy import ndimage


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)



def nan_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.nanpercentile(x, q))



def sanitize_nonfinite_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X
    if X.ndim == 1:
        X = X[:, None]
        squeeze_back = True
    else:
        squeeze_back = False
    bad = ~np.isfinite(X)
    if np.any(bad):
        med = np.nanmedian(X, axis=0)
        med[~np.isfinite(med)] = 0.0
        for j in range(X.shape[1]):
            col_bad = bad[:, j]
            if np.any(col_bad):
                X[col_bad, j] = med[j]
    if squeeze_back:
        return X[:, 0]
    return X



def smooth_moving_average(x: np.ndarray, span: int = 3, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if span <= 1 or x.size == 0:
        return x.copy()
    kernel_shape = [1] * x.ndim
    kernel_shape[axis] = span
    kernel = np.ones(kernel_shape, dtype=float)
    y = signal.convolve(x, kernel, mode="same")
    counts = signal.convolve(np.ones_like(x, dtype=float), kernel, mode="same")
    counts[counts == 0] = 1.0
    return y / counts



def medfilt1(x: np.ndarray, kernel_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    if x.ndim == 1:
        return ndimage.median_filter(x, size=kernel_size, mode="nearest")
    size = [1] * x.ndim
    size[0] = kernel_size
    return ndimage.median_filter(x, size=size, mode="nearest")



def binary_auc(scores: np.ndarray, y: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(y).reshape(-1)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :]).astype(float)
    return float(np.nanmean(comparisons))



def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.zeros_like(x)
    valid = np.isfinite(x)
    if not np.any(valid):
        return y
    xv = x[valid]
    med = float(np.nanmedian(xv))
    mad = float(np.nanmedian(np.abs(xv - med)))
    if not np.isfinite(mad) or mad < 1e-8:
        mad = float(np.nanstd(xv))
    if not np.isfinite(mad) or mad < 1e-8:
        mad = 1.0
    y[valid] = (xv - med) / mad
    return y



def pdist_local(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    vals = []
    for i in range(n - 1):
        diffs = X[i + 1 :, :] - X[i, :]
        vals.append(np.sqrt(np.sum(diffs * diffs, axis=1)))
    if not vals:
        return np.zeros((0,), dtype=float)
    return np.concatenate(vals)



def squareform_local(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size == 0:
        return np.zeros((1, 1), dtype=float)
    n = int(round((1 + math.sqrt(1 + 8 * v.size)) / 2))
    D = np.zeros((n, n), dtype=float)
    idx = 0
    for i in range(n - 1):
        length = n - i - 1
        D[i, i + 1 :] = v[idx : idx + length]
        D[i + 1 :, i] = v[idx : idx + length]
        idx += length
    return D



def matlab_cell(values: list[Any]) -> np.ndarray:
    arr = np.empty((len(values), 1), dtype=object)
    for i, v in enumerate(values):
        arr[i, 0] = v
    return arr



def to_serializable(obj: Any) -> Any:
    if obj is None:
        return np.array([], dtype=float)
    if is_dataclass(obj):
        out: dict[str, Any] = {}
        for k, v in asdict(obj).items():
            if v is None:
                continue
            out[str(k)] = to_serializable(v)
        return out
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[str(k)] = to_serializable(v)
        return out
    if isinstance(obj, (list, tuple)):
        if not obj:
            return np.empty((0, 1), dtype=object)
        if all(isinstance(v, (str, bytes)) for v in obj):
            return np.array(list(obj), dtype=object).reshape(-1, 1)
        if all(not isinstance(v, (list, tuple, dict)) and not is_dataclass(v) for v in obj):
            try:
                return np.array(obj)
            except Exception:
                pass
        return matlab_cell([to_serializable(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj



def ensure_numpy(x: Any, dtype: Any = float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)
