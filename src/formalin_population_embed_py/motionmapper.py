from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage, optimize, signal, spatial
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from skimage import measure, morphology, segmentation

from .utils import sanitize_nonfinite_matrix



def morlet_conj_ft(w: np.ndarray, omega0: float) -> np.ndarray:
    return np.pi ** (-0.25) * np.exp(-0.5 * (w - omega0) ** 2)



def fast_wavelet_morlet_convolution(x: np.ndarray, f: np.ndarray, omega0: float, dt: float) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.asarray(x, dtype=float).reshape(-1)
    N0 = x.size
    if N0 == 0:
        return np.zeros((len(f), 0), dtype=float), None
    test_odd = N0 % 2 == 1
    if test_odd:
        x = np.concatenate([x, np.zeros(1, dtype=float)])
    M = x.size
    x = np.concatenate([np.zeros(M // 2, dtype=float), x, np.zeros(M // 2, dtype=float)])
    N = x.size
    scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * np.asarray(f, dtype=float))
    omegavals = 2 * np.pi * np.arange(-N / 2, N / 2, 1.0) / (N * dt)
    x_hat = np.fft.fftshift(np.fft.fft(x))
    if test_odd:
        idx = np.arange(M // 2, M // 2 + M - 1, dtype=int)
    else:
        idx = np.arange(M // 2, M // 2 + M, dtype=int)
    amp = np.zeros((len(f), idx.size), dtype=float)
    W_out = None
    const_base = np.pi ** (-0.25) * np.exp(0.25 * (omega0 - np.sqrt(omega0**2 + 2)) ** 2)
    for i, scale in enumerate(scales):
        m = morlet_conj_ft(-omegavals * scale, omega0)
        q = np.fft.ifft(m * x_hat) * np.sqrt(scale)
        q = q[idx]
        amp[i, :] = np.abs(q) * const_base / np.sqrt(2 * scale)
    return amp, W_out



def find_wavelets(projections: np.ndarray, num_modes: int, parameters: Any) -> tuple[np.ndarray, np.ndarray]:
    projections = np.asarray(projections, dtype=float)
    L = projections.shape[1]
    if num_modes is None or num_modes > L:
        num_modes = L
    omega0 = float(parameters.omega0)
    num_periods = int(parameters.num_periods)
    dt = 1.0 / float(parameters.sampling_freq)
    minT = 1.0 / float(parameters.max_f)
    maxT = 1.0 / float(parameters.min_f)
    Ts = minT * 2 ** ((np.arange(num_periods) * np.log(maxT / minT) / np.log(2)) / (num_periods - 1))
    f = np.flip(1.0 / Ts)
    N = projections.shape[0]
    amplitudes = np.zeros((N, num_modes * num_periods), dtype=float)
    for i in range(num_modes):
        amp, _ = fast_wavelet_morlet_convolution(projections[:, i], f, omega0, dt)
        amplitudes[:, np.arange(num_periods) + i * num_periods] = amp.T
    return amplitudes, f



def run_tsne_with_config(X: np.ndarray, perplexity: float | None, seed: int = 0, backend: str = "pca") -> np.ndarray:
    X = sanitize_nonfinite_matrix(np.asarray(X, dtype=float))
    if X.shape[0] <= 1:
        if X.shape[0] == 0:
            return np.zeros((0, 2), dtype=float)
        return np.zeros((1, 2), dtype=float)
    backend = str(backend).strip().lower()
    if backend == "sklearn":
        if perplexity is None or not np.isfinite(perplexity):
            perp = min(30.0, max(5.0, (X.shape[0] - 1) / 3.0))
        else:
            perp = min(float(perplexity), max(5.0, (X.shape[0] - 1) / 3.0))
        tsne_kwargs = dict(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=int(seed),
            metric="euclidean",
            verbose=0,
        )
        try:
            tsne = TSNE(max_iter=500, **tsne_kwargs)
        except TypeError:
            tsne = TSNE(n_iter=500, **tsne_kwargs)
        return tsne.fit_transform(X)
    if backend == "pca":
        n_comp = min(2, X.shape[0], X.shape[1])
        if n_comp <= 0:
            return np.zeros((X.shape[0], 2), dtype=float)
        Y = PCA(n_components=n_comp, svd_solver="full").fit_transform(X)
        if Y.shape[1] == 1:
            Y = np.concatenate([Y, np.zeros((Y.shape[0], 1), dtype=float)], axis=1)
        sd = np.nanstd(Y, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        return Y[:, :2] / sd.reshape(1, -1)
    raise ValueError(f"Unsupported tsne backend {backend!r}")



def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        step = 1.0
        return np.array([centers[0] - step / 2, centers[0] + step / 2], dtype=float)
    mids = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])



def find_point_density(points: np.ndarray, sigma: float, num_points: int | tuple[int, int] = 1001, range_vals: list[float] | tuple[float, ...] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if isinstance(num_points, int):
        nx = ny = int(num_points)
    else:
        nx, ny = int(num_points[0]), int(num_points[1])
    if nx % 2 == 0:
        nx += 1
    if ny % 2 == 0:
        ny += 1
    if range_vals is None:
        range_vals = (-110.0, 110.0)
    if len(range_vals) == 2:
        x = np.linspace(range_vals[0], range_vals[1], nx)
        y = x.copy()
    else:
        x = np.linspace(range_vals[0], range_vals[1], nx)
        y = np.linspace(range_vals[2], range_vals[3], ny)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    G = np.exp(-0.5 * (XX**2 + YY**2) / sigma**2) / (2 * np.pi * sigma**2)
    x_edges = _centers_to_edges(x)
    y_edges = _centers_to_edges(y)
    Z, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_edges, y_edges])
    Z = Z / max(np.sum(Z), 1.0)
    density = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(G) * np.fft.fft2(Z.T)))).T
    density[density < 0] = 0
    return x, density, G, Z



def _grid_index_from_points(points: np.ndarray, xx: np.ndarray) -> np.ndarray:
    max_x = float(np.max(xx))
    vals = np.round((points + max_x) * len(xx) / (2 * max_x)).astype(int)
    vals[vals < 1] = 1
    vals[vals > len(xx)] = len(xx)
    return vals - 1



def return_templates(
    y_data: np.ndarray,
    signal_data: np.ndarray,
    min_template_length: int = 10,
    kd_neighbors: int = 10,
    plots_on: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    del plots_on
    y_data = np.asarray(y_data, dtype=float)
    signal_data = np.asarray(signal_data, dtype=float)
    max_y = int(np.ceil(np.max(np.abs(y_data)))) + 1 if y_data.size else 1
    d = signal_data.shape[1] if signal_data.ndim == 2 and signal_data.size else 1
    n_neighbors = int(min(kd_neighbors + 1, max(1, y_data.shape[0])))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(y_data)
    distances, _ = nn.kneighbors(y_data)
    kth_col = min(kd_neighbors, distances.shape[1] - 1)
    sigma = float(np.nanmedian(distances[:, kth_col])) if distances.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    xx, density, _, _ = find_point_density(y_data, sigma, 501, [-max_y, max_y])
    L = segmentation.watershed(-density, connectivity=2)
    vals = _grid_index_from_points(y_data, xx)
    watershed_values = L[vals[:, 1], vals[:, 0]]
    max_L = int(np.max(L)) if L.size else 0
    templates = [signal_data[watershed_values == i, :] for i in range(1, max_L + 1)]
    lengths = np.array([tpl.shape[0] / d for tpl in templates], dtype=float)
    keep = np.flatnonzero(lengths >= min_template_length)
    vals2 = np.zeros_like(watershed_values, dtype=int)
    for new_i, old_i in enumerate(keep, start=1):
        vals2[watershed_values == (old_i + 1)] = new_i
    templates_keep = [templates[i] for i in keep]
    lengths_keep = lengths[keep] if keep.size else np.zeros((0,), dtype=float)
    return templates_keep, xx, density, sigma, lengths_keep, L, vals2



def find_templates_from_data(
    signal_data: np.ndarray,
    y_data: np.ndarray,
    signal_amps: np.ndarray,
    num_per_dataset: int,
    parameters: Any,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    kd_neighbors = int(getattr(parameters, "kd_neighbors", 5))
    min_template_length = int(getattr(parameters, "min_template_length", 1))
    templates, _, _, _, template_lengths, _, vals = return_templates(y_data, signal_data, min_template_length, kd_neighbors, False)
    N = len(templates)
    d = signal_data.shape[1]
    if N == 0:
        take = min(signal_data.shape[0], num_per_dataset)
        return signal_data[:take, :], signal_amps[:take]
    selected_data = np.zeros((num_per_dataset, d), dtype=float)
    selected_amps = np.zeros((num_per_dataset,), dtype=float)
    num_in_group = np.round(num_per_dataset * template_lengths / np.sum(template_lengths)).astype(int)
    num_in_group[num_in_group == 0] = 1
    rng = np.random.default_rng(seed)
    sum_val = int(np.sum(num_in_group))
    if sum_val < num_per_dataset:
        q = num_per_dataset - sum_val
        idx = rng.choice(N, size=q, replace=True)
        for j in idx:
            num_in_group[j] += 1
    elif sum_val > num_per_dataset:
        q = sum_val - num_per_dataset
        idx2 = np.flatnonzero(num_in_group > 1)
        if idx2.size < q:
            idx2 = np.arange(num_in_group.size)
        idx = rng.choice(idx2, size=q, replace=False if idx2.size >= q else True)
        for j in idx:
            if num_in_group[j] > 1:
                num_in_group[j] -= 1
    cum = np.concatenate([[0], np.cumsum(num_in_group)])
    for j in range(N):
        amps = signal_amps[vals == (j + 1)]
        tpl = templates[j]
        if tpl.shape[0] == 0:
            continue
        need = int(num_in_group[j])
        if tpl.shape[0] >= need:
            idx2 = rng.choice(tpl.shape[0], size=need, replace=False)
        else:
            idx2 = rng.choice(tpl.shape[0], size=need, replace=True)
        selected_data[cum[j] : cum[j + 1], :] = tpl[idx2, :]
        if amps.size == 0:
            selected_amps[cum[j] : cum[j + 1]] = 1.0
        else:
            if amps.size >= need:
                amp_idx = rng.choice(amps.size, size=need, replace=False)
            else:
                amp_idx = rng.choice(amps.size, size=need, replace=True)
            selected_amps[cum[j] : cum[j + 1]] = amps[amp_idx]
    return selected_data, selected_amps



def return_correct_sigma_sparse(ds: np.ndarray, perplexity: float = 32, tol: float = 1e-5, max_neighbors: int = 200) -> tuple[float, np.ndarray]:
    ds = np.asarray(ds, dtype=float).reshape(-1)
    high_guess = float(np.max(ds)) if ds.size else 1.0
    low_guess = 1e-10
    sigma = 0.5 * (high_guess + low_guess)
    sort_idx = np.argsort(ds)
    keep_idx = sort_idx[: min(max_neighbors, ds.size)]
    ds_keep = ds[keep_idx]

    def compute_p(sig: float) -> tuple[np.ndarray, float, float]:
        p = np.exp(-0.5 * ds_keep**2 / sig**2)
        s = np.sum(p)
        if s == 0 or not np.isfinite(s):
            p = np.ones_like(ds_keep) / max(ds_keep.size, 1)
        else:
            p = p / s
        idx = p > 0
        H = np.sum(-p[idx] * np.log(p[idx]) / np.log(2))
        P = 2**H
        return p, H, P

    p, _, P = compute_p(sigma)
    si = 0
    while abs(P - perplexity) >= tol and si < 200:
        if P > perplexity:
            high_guess = sigma
        else:
            low_guess = sigma
        sigma = 0.5 * (high_guess + low_guess)
        p, _, P = compute_p(sigma)
        si += 1
    full_p = np.zeros(ds.size, dtype=float)
    full_p[keep_idx] = p
    return sigma, full_p



def calculate_kl_cost(x: np.ndarray, ydata: np.ndarray, ps: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(1, -1)
    ydata = np.asarray(ydata, dtype=float)
    ps = np.asarray(ps, dtype=float).reshape(-1)
    d = np.sqrt(np.sum((ydata - x) ** 2, axis=1))
    d2 = d**2
    return float(np.log(np.sum((1 + d2) ** -1)) + np.sum(ps * np.log(1 + d2)))



def _point_in_convex_hull(point: np.ndarray, hull_points: np.ndarray) -> bool:
    if hull_points.shape[0] < 3:
        return False
    try:
        hull = spatial.ConvexHull(hull_points)
        poly = hull_points[hull.vertices, :]
        return bool(MplPath(poly).contains_point(point))
    except Exception:
        return False



def find_tdistributed_projections_fmin(
    data: np.ndarray,
    signal_data: np.ndarray,
    y_data: np.ndarray,
    perplexity: float | None,
    parameters: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = sanitize_nonfinite_matrix(np.asarray(data, dtype=float))
    signal_data = sanitize_nonfinite_matrix(np.asarray(signal_data, dtype=float))
    y_data = np.asarray(y_data, dtype=float)
    batch_size = int(getattr(parameters, "batch_size", 10000))
    max_neighbors = int(getattr(parameters, "max_neighbors", 200))
    sigma_tolerance = float(getattr(parameters, "sigma_tolerance", 1e-5))
    if perplexity is None or not np.isfinite(perplexity):
        perplexity = 32.0
    N = data.shape[0]
    z_values = np.zeros((N, 2), dtype=float)
    z_guesses = np.zeros((N, 2), dtype=float)
    z_costs = np.zeros(N, dtype=float)
    in_conv_hull = np.zeros(N, dtype=bool)
    mean_max = np.zeros(N, dtype=float)
    exit_flags = np.zeros(N, dtype=float)
    batches = int(np.ceil(N / batch_size)) if batch_size > 0 else 1
    for j in range(batches):
        idx = np.arange(j * batch_size, min((j + 1) * batch_size, N))
        current_data = data[idx, :]
        D2 = cdist(current_data, signal_data)
        current_guesses = np.zeros((idx.size, 2), dtype=float)
        current = np.zeros((idx.size, 2), dtype=float)
        t_costs = np.zeros(idx.size, dtype=float)
        current_poly = np.zeros(idx.size, dtype=bool)
        current_mean_max = np.zeros(idx.size, dtype=float)
        for i in range(idx.size):
            _, p = return_correct_sigma_sparse(D2[i, :], perplexity, sigma_tolerance, max_neighbors)
            idx2 = p > 0
            z = y_data[idx2, :]
            p_nz = p[idx2]
            if z.shape[0] == 0:
                z = y_data
                p_nz = np.ones(y_data.shape[0], dtype=float) / max(y_data.shape[0], 1)
            max_idx = int(np.argmax(p))
            a = np.sum(z * p_nz.reshape(-1, 1), axis=0)
            guesses = np.vstack([a, y_data[max_idx, :]])
            guess_costs = np.array([calculate_kl_cost(guesses[g, :], z, p_nz) for g in range(2)], dtype=float)
            mI = int(np.argmin(guess_costs))
            best_point = guesses[mI, :].copy()
            best_cost = float(guess_costs[mI])
            best_flag = 0.0
            refine_projection = bool(getattr(parameters, "projection_refine", False))
            max_iter = int(getattr(parameters, "max_optim_iter", 25))
            if refine_projection and max_iter > 0:
                try:
                    res = optimize.minimize(
                        lambda x: calculate_kl_cost(x, z, p_nz),
                        best_point,
                        method="L-BFGS-B",
                        options={"maxiter": max_iter, "ftol": 1e-6},
                    )
                    if np.isfinite(res.fun) and float(res.fun) <= best_cost:
                        best_point = np.asarray(res.x, dtype=float)
                        best_cost = float(res.fun)
                        best_flag = 1.0 if bool(res.success) else -1.0
                except Exception:
                    pass
            current_poly[i] = _point_in_convex_hull(best_point, z)
            exit_flags[idx[i]] = best_flag
            current_guesses[i, :] = guesses[mI, :]
            current[i, :] = best_point
            t_costs[i] = best_cost
            current_mean_max[i] = mI + 1
        z_guesses[idx, :] = current_guesses
        z_values[idx, :] = current
        z_costs[idx] = t_costs
        in_conv_hull[idx] = current_poly
        mean_max[idx] = current_mean_max
    return z_values, z_costs, z_guesses, in_conv_hull, mean_max, exit_flags



def gaussianfilterdata_derivative_zeropad(x: np.ndarray, sigma: float, dt: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n == 0:
        return x.copy()
    if sigma <= 0:
        y = np.concatenate([[0.0], np.diff(x) / dt])
        return y
    sigma_samples = max(float(sigma) / float(dt), np.finfo(float).eps)
    half_width = max(1, int(np.ceil(4 * sigma_samples)))
    t = np.arange(-half_width, half_width + 1, dtype=float)
    g = np.exp(-0.5 * (t / sigma_samples) ** 2)
    g = g / np.sum(g)
    dg = -(t / (sigma_samples**2)) * g
    dg = dg / dt
    y = signal.convolve(x, dg, mode="same")
    if y.shape[0] != n:
        if y.shape[0] > n:
            start = (y.shape[0] - n) // 2
            y = y[start : start + n]
        else:
            y = np.pad(y, (0, n - y.shape[0]), mode="constant")
    return y



def _large_bw_conn_comp(x: np.ndarray, min_length: int, vals: np.ndarray | None = None, min_val: float | None = None) -> list[np.ndarray]:
    x = np.asarray(x).astype(bool).reshape(-1)
    if x.size == 0:
        return []
    labels, num = ndimage.label(x)
    comps = []
    for i in range(1, num + 1):
        pix = np.flatnonzero(labels == i)
        if pix.size >= min_length:
            if vals is not None and min_val is not None:
                if np.max(vals[pix]) >= min_val:
                    comps.append(pix)
            else:
                comps.append(pix)
    return comps



def find_watershed_regions_v2(
    all_z: np.ndarray,
    xx: np.ndarray,
    LL: np.ndarray,
    v_smooth: float = 1.0,
    median_length: int = 1,
    p_threshold: list[float] | tuple[float, float] | float | None = None,
    min_rest: int = 5,
    obj: Any | None = None,
    fit_only: bool = False,
) -> tuple[np.ndarray, list[list[np.ndarray]], np.ndarray, Any, np.ndarray, np.ndarray]:
    if p_threshold is None:
        p_threshold = [0.67, 0.33]
    all_z = np.asarray(all_z, dtype=float)
    if all_z.ndim != 2 or all_z.shape[1] != 2:
        raise ValueError("all_z must be Nx2")
    rest_length = 10
    dt = 0.01
    num_to_test = 50000
    N = all_z.shape[0]
    smooth_z = all_z.copy()
    if median_length > 0:
        smooth_z[:, 0] = signal.medfilt(all_z[:, 0], kernel_size=max(1, int(median_length)) | 1)
        smooth_z[:, 1] = signal.medfilt(all_z[:, 1], kernel_size=max(1, int(median_length)) | 1)
    vx = gaussianfilterdata_derivative_zeropad(smooth_z[:, 0], v_smooth, dt)
    vy = gaussianfilterdata_derivative_zeropad(smooth_z[:, 1], v_smooth, dt)
    v = np.sqrt(vx**2 + vy**2)
    lv = np.log10(np.maximum(v, np.finfo(float).tiny)).reshape(-1, 1)
    if obj is None:
        sample_idx = np.linspace(0, max(0, lv.shape[0] - 1), min(num_to_test, lv.shape[0]), dtype=int)
        sample = lv[sample_idx, :]
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(sample)
        obj = gmm
    posts = obj.predict_proba(lv)
    max_idx = int(np.argmax(obj.means_.reshape(-1)))
    p_rest = 1.0 - posts[:, max_idx]
    if fit_only:
        return np.zeros((0,), dtype=int), [], v, obj, p_rest, np.zeros((0, 2), dtype=int)
    vals = _grid_index_from_points(smooth_z, xx)
    watershed_values = LL[vals[:, 1], vals[:, 0]].astype(int)
    diff_values = np.concatenate([[False], np.abs(np.diff(watershed_values)) == 0])
    if np.isscalar(p_threshold):
        CC = _large_bw_conn_comp((p_rest > float(p_threshold)) & diff_values, min_rest)
    else:
        p_threshold_arr = np.asarray(p_threshold, dtype=float).reshape(-1)
        min_val = float(np.min(p_threshold_arr))
        max_val = float(np.max(p_threshold_arr))
        comps = _large_bw_conn_comp((p_rest > min_val) & diff_values, min_rest)
        CC = [pix for pix in comps if np.max(p_rest[pix]) >= max_val]
    L = int(np.max(LL)) if LL.size else 0
    watershed_regions = np.zeros(N, dtype=int)
    for pix in CC:
        segment_assignment = int(np.bincount(watershed_values[pix]).argmax()) if pix.size else 0
        watershed_regions[pix] = segment_assignment
    for i in range(1, L + 1):
        comps = _large_bw_conn_comp(watershed_values == i, rest_length)
        for pix in comps:
            watershed_regions[pix] = i
    segments: list[list[np.ndarray]] = []
    for i in range(1, L + 1):
        labels, num = ndimage.label(watershed_regions == i)
        pix_list = [np.flatnonzero(labels == j) for j in range(1, num + 1)]
        segments.append(pix_list)
    return watershed_regions, segments, v, obj, p_rest, vals



def build_density_watershed(
    Y_training: np.ndarray,
    sigma_density: float = 0.8,
    grid_size: int = 501,
) -> dict[str, Any]:
    y_min = np.min(Y_training, axis=0)
    y_max = np.max(Y_training, axis=0)
    max_abs = max(abs(y_min[0]), abs(y_max[0]), abs(y_min[1]), abs(y_max[1])) * 1.05
    xx, d, _, _ = find_point_density(Y_training, sigma_density, grid_size, [-max_abs, max_abs])
    D = d.copy()
    positive_vals = D[D > 0]
    density_floor = float(np.nanpercentile(positive_vals, 5)) if positive_vals.size else 0.0
    LL = segmentation.watershed(-d, connectivity=2)
    LL2 = LL.copy().astype(int)
    if density_floor > 0:
        bg_mask = D < density_floor
        LL2[bg_mask] = -1
        D[bg_mask] = 0
    LL2[d < 1e-6] = -1
    fg_mask = LL2 > 0
    display_mask = fg_mask.copy()
    boundary_trim_pixels = 4
    if boundary_trim_pixels > 0:
        display_mask = morphology.erosion(display_mask.astype(np.uint8), morphology.disk(boundary_trim_pixels)) > 0
        display_mask = ndimage.binary_fill_holes(display_mask)
        if not np.any(display_mask):
            display_mask = fg_mask
    contours = measure.find_contours(display_mask.astype(float), 0.5)
    boundary_polys = []
    for poly in contours:
        if poly.shape[0] >= 5:
            poly[:, 0] = ndimage.uniform_filter1d(poly[:, 0], size=5, mode="nearest")
            poly[:, 1] = ndimage.uniform_filter1d(poly[:, 1], size=5, mode="nearest")
        poly[:, 0] = np.clip(poly[:, 0], 0, LL2.shape[0] - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, LL2.shape[1] - 1)
        boundary_polys.append(poly)
    llbwb = np.vstack(boundary_polys) if boundary_polys else np.zeros((0, 2), dtype=float)
    return {"D": D, "LL": LL, "LL2": LL2, "llbwb": llbwb, "boundaryPolys": boundary_polys, "xx": xx}
