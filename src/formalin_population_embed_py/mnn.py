from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .utils import sanitize_nonfinite_matrix



def _ensure_knn_matrix(idx: np.ndarray, n_rows: int, k: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return np.ones((n_rows, max(1, k)), dtype=int)
    if idx.ndim == 1:
        idx = idx[:, None]
    if idx.shape[1] < k:
        pad = np.repeat(idx[:, -1:], k - idx.shape[1], axis=1)
        idx = np.concatenate([idx, pad], axis=1)
    return idx



def _knn_indices(ref: np.ndarray, query: np.ndarray, k: int, distance: str = "euclidean") -> np.ndarray:
    k = int(min(max(1, k), ref.shape[0]))
    nn = NearestNeighbors(n_neighbors=k, metric=distance)
    nn.fit(ref)
    _, idx = nn.kneighbors(query)
    return idx



def _maxrows_local(mat: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.lexsort((-mat[:, 1], -mat[:, 0]))
    best_idx = int(order[0])
    return mat[best_idx, :], best_idx



def _select_reference_indices(state_ref: np.ndarray, anchor_ref: np.ndarray, target_state: int, opts: dict[str, Any]) -> np.ndarray:
    state_idx = np.flatnonzero(state_ref == target_state)
    if state_idx.size == 0:
        return state_idx
    anchor_idx = state_idx[anchor_ref[state_idx]]
    if opts.get("preferAnchorReference", True) and anchor_idx.size >= opts.get("minStatePoints", max(5, opts.get("k", 20))):
        return anchor_idx
    if state_idx.size >= opts.get("minStatePoints", max(5, opts.get("k", 20))):
        return state_idx
    if anchor_idx.size > 0:
        return anchor_idx
    return state_idx



def _compute_offsets_basic(
    Z_query: np.ndarray,
    X0_query: np.ndarray,
    Z_ref: np.ndarray,
    X0_ref: np.ndarray,
    opts: dict[str, Any],
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_query = Z_query.shape[0]
    offsets = np.zeros((n_query, X0_query.shape[1]), dtype=float)
    used = np.zeros(n_query, dtype=bool)
    if Z_query.size == 0 or Z_ref.size == 0:
        return offsets, used
    k_ref = int(min(opts.get("k", 20), Z_ref.shape[0]))
    k_query = int(min(opts.get("k", 20), Z_query.shape[0]))
    idx_q2r = _ensure_knn_matrix(_knn_indices(Z_ref, Z_query, k_ref, opts.get("distance", "euclidean")), Z_query.shape[0], k_ref)
    idx_r2q = _ensure_knn_matrix(_knn_indices(Z_query, Z_ref, k_query, opts.get("distance", "euclidean")), Z_ref.shape[0], k_query)
    for i in range(n_query):
        anchors = idx_q2r[i, :]
        mutual_mask = np.array([np.any(idx_r2q[a, :] == i) for a in anchors], dtype=bool)
        anchors = anchors[mutual_mask]
        if anchors.size == 0:
            anchors = idx_q2r[i, :1]
        else:
            used[i] = True
        D = np.sum((Z_ref[anchors, :] - Z_query[i, :]) ** 2, axis=1)
        w = np.exp(-D / (2 * sigma2))
        if not np.any(np.isfinite(w)) or np.sum(w) == 0:
            w = np.ones(anchors.size, dtype=float)
        w = w.reshape(-1) / np.sum(w)
        delta = w @ (X0_ref[anchors, :] - X0_query[i, :])
        offsets[i, :] = delta
    return offsets, used



def _compute_guided_offsets(
    Z_query: np.ndarray,
    X0_query: np.ndarray,
    state_query: np.ndarray,
    Z_ref: np.ndarray,
    X0_ref: np.ndarray,
    state_ref: np.ndarray,
    anchor_ref: np.ndarray,
    opts: dict[str, Any],
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_query = Z_query.shape[0]
    offsets = np.zeros((n_query, X0_query.shape[1]), dtype=float)
    used = np.zeros(n_query, dtype=bool)
    processed = np.zeros(n_query, dtype=bool)
    if opts.get("stateAware", False):
        for s in np.unique(state_query):
            if s <= 1:
                continue
            q_idx = np.flatnonzero(state_query == s)
            if q_idx.size == 0:
                continue
            ref_idx = _select_reference_indices(state_ref, anchor_ref, int(s), opts)
            if ref_idx.size < opts.get("minStatePoints", max(5, opts.get("k", 20))):
                continue
            offsets_state, used_state = _compute_offsets_basic(Z_query[q_idx, :], X0_query[q_idx, :], Z_ref[ref_idx, :], X0_ref[ref_idx, :], opts, sigma2)
            offsets[q_idx, :] = offsets_state
            used[q_idx] = used_state
            processed[q_idx] = True
    fallback_idx = np.flatnonzero(~processed)
    if fallback_idx.size > 0:
        offsets_fallback, used_fallback = _compute_offsets_basic(Z_query[fallback_idx, :], X0_query[fallback_idx, :], Z_ref, X0_ref, opts, sigma2)
        offsets[fallback_idx, :] = offsets_fallback
        used[fallback_idx] = used_fallback
    return offsets, used



def fit_guided_mnn_correction(
    X: np.ndarray,
    batch_vec: np.ndarray,
    opts: dict[str, Any] | None,
    state_vec: np.ndarray | None,
    anchor_mask: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    if opts is None:
        opts = {}
    if state_vec is None:
        state_vec = np.ones(X.shape[0], dtype=int)
    if anchor_mask is None:
        anchor_mask = np.zeros(X.shape[0], dtype=bool)
    opts = {
        "k": int(opts.get("k", 20)),
        "ndim": int(opts.get("ndim", min(50, X.shape[1]))),
        "sigma": opts.get("sigma", None),
        "distance": opts.get("distance", "euclidean"),
        "verbose": bool(opts.get("verbose", True)),
        "stateAware": bool(opts.get("stateAware", False)),
        "preferAnchorReference": bool(opts.get("preferAnchorReference", True)),
        "minStatePoints": int(opts.get("minStatePoints", max(5, int(opts.get("k", 20))))),
    }
    X = np.asarray(X, dtype=float)
    batch_vec = np.asarray(batch_vec, dtype=float).reshape(-1)
    state_vec = np.asarray(state_vec, dtype=int).reshape(-1)
    anchor_mask = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    ndim = int(min(opts["ndim"], X.shape[1], X.shape[0]))
    mu = np.nanmedian(X, axis=0)
    mad0 = 1.4826 * np.nanmedian(np.abs(X - mu), axis=0)
    mad0[~np.isfinite(mad0) | (mad0 == 0)] = 1.0
    Xs = (X - mu) / mad0
    pc_mu = np.nanmean(Xs, axis=0)
    X0 = sanitize_nonfinite_matrix(Xs - pc_mu)
    _, _, Vt = np.linalg.svd(X0, full_matrices=False)
    W = Vt.T[:, :ndim]
    Z = X0 @ W
    groups, gidx = np.unique(batch_vec, return_inverse=True)
    Ns = np.bincount(gidx)
    if opts["stateAware"] and np.any(anchor_mask):
        anchor_counts = np.bincount(gidx, weights=anchor_mask.astype(float), minlength=groups.size)
        rank_mat = np.column_stack([anchor_counts, Ns])
        _, ref_i = _maxrows_local(rank_mat)
    else:
        ref_i = int(np.argmax(Ns))
    ref_batch = groups[ref_i]
    ref_mask = gidx == ref_i
    X0_ref = X0[ref_mask, :]
    Z_ref = Z[ref_mask, :]
    if opts["sigma"] is None:
        k_sigma = int(min(max(3, opts["k"]), max(1, Z_ref.shape[0])))
        idx_ref = _ensure_knn_matrix(_knn_indices(Z_ref, Z_ref, k_sigma, opts["distance"]), Z_ref.shape[0], k_sigma)
        dd = np.zeros(Z_ref.shape[0], dtype=float)
        for i in range(Z_ref.shape[0]):
            nbr = idx_ref[i, -1]
            dd[i] = np.linalg.norm(Z_ref[i, :] - Z_ref[nbr, :])
        sigma = float(np.nanmedian(dd[dd > 0])) if np.any(dd > 0) else 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
    else:
        sigma = float(opts["sigma"])
    sigma2 = sigma**2
    X0_corr = np.zeros_like(X0)
    X0_corr[ref_mask, :] = X0_ref
    order = [ref_i] + [i for i in range(groups.size) if i != ref_i]
    for oi in range(1, len(order)):
        b = order[oi]
        mask_b = gidx == b
        if not np.any(mask_b):
            continue
        batch_now_mask = np.isin(gidx, order[:oi])
        Z_ref_now = X0_corr[batch_now_mask, :] @ W
        X0_ref_now = X0_corr[batch_now_mask, :]
        state_ref_now = state_vec[batch_now_mask]
        anchor_ref_now = anchor_mask[batch_now_mask]
        Z_b = Z[mask_b, :]
        X0_b = X0[mask_b, :]
        state_b = state_vec[mask_b]
        offsets, _ = _compute_guided_offsets(Z_b, X0_b, state_b, Z_ref_now, X0_ref_now, state_ref_now, anchor_ref_now, opts, sigma2)
        X0_corr[mask_b, :] = X0_b + offsets
    Xcorr = (X0_corr + pc_mu) * mad0 + mu
    model = {
        "enabled": True,
        "mode": "guided_mnn",
        "mu": mu,
        "mad": mad0,
        "pcMu": pc_mu,
        "W": W,
        "ndim": ndim,
        "k": opts["k"],
        "sigma": sigma,
        "distance": opts["distance"],
        "stateAware": opts["stateAware"],
        "preferAnchorReference": opts["preferAnchorReference"],
        "minStatePoints": opts["minStatePoints"],
        "refZ": X0_corr @ W,
        "refX0": X0_corr,
        "refStateVec": state_vec,
        "refAnchorMask": anchor_mask,
        "referenceBatch": ref_batch,
        "referenceBatchIndex": ref_i + 1,
        "batchVec": batch_vec,
        "note": "Guided MNN correction",
    }
    return model, Xcorr



def apply_guided_mnn_correction(
    Xnew: np.ndarray,
    model: dict[str, Any] | None,
    state_vec: np.ndarray | None,
    anchor_mask: np.ndarray | None,
) -> np.ndarray:
    if model is None or not isinstance(model, dict) or not model.get("enabled", False):
        return np.asarray(Xnew, dtype=float)
    if state_vec is None:
        state_vec = np.ones(Xnew.shape[0], dtype=int)
    if anchor_mask is None:
        anchor_mask = np.zeros(Xnew.shape[0], dtype=bool)
    Xnew = np.asarray(Xnew, dtype=float)
    state_vec = np.asarray(state_vec, dtype=int).reshape(-1)
    _ = np.asarray(anchor_mask, dtype=bool).reshape(-1)
    Xs = (Xnew - model["mu"]) / model["mad"]
    X0 = sanitize_nonfinite_matrix(Xs - model["pcMu"])
    Z = X0 @ model["W"]
    sigma2 = float(model["sigma"]) ** 2
    opts = {
        "k": int(model["k"]),
        "distance": model["distance"],
        "stateAware": bool(model.get("stateAware", False)),
        "preferAnchorReference": bool(model.get("preferAnchorReference", False)),
        "minStatePoints": int(model.get("minStatePoints", max(5, int(model["k"])))),
    }
    offsets = np.zeros((Z.shape[0], X0.shape[1]), dtype=float)
    processed = np.zeros(Z.shape[0], dtype=bool)
    if opts["stateAware"] and ("refStateVec" in model) and ("refAnchorMask" in model):
        for s in np.unique(state_vec):
            if s <= 1:
                continue
            q_idx = np.flatnonzero(state_vec == s)
            if q_idx.size == 0:
                continue
            ref_idx = _select_reference_indices(np.asarray(model["refStateVec"], dtype=int), np.asarray(model["refAnchorMask"], dtype=bool), int(s), opts)
            if ref_idx.size < opts["minStatePoints"]:
                continue
            offsets_state, _ = _compute_offsets_basic(Z[q_idx, :], X0[q_idx, :], np.asarray(model["refZ"])[ref_idx, :], np.asarray(model["refX0"])[ref_idx, :], opts, sigma2)
            offsets[q_idx, :] = offsets_state
            processed[q_idx] = True
    fallback_idx = np.flatnonzero(~processed)
    if fallback_idx.size > 0:
        offsets_fallback, _ = _compute_offsets_basic(
            Z[fallback_idx, :], X0[fallback_idx, :], np.asarray(model["refZ"]), np.asarray(model["refX0"]), opts, sigma2
        )
        offsets[fallback_idx, :] = offsets_fallback
    X0_corr = X0 + offsets
    Xcorr = (X0_corr + model["pcMu"]) * model["mad"] + model["mu"]
    return Xcorr
