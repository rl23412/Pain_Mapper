from __future__ import annotations

import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import create_analysis_outputs
from .config import PipelineConfig, SupervisionConfig
from .features import (
    compute_joint_velocity_features,
    compute_smoothed_pairwise_distances,
    compute_z_coordinate_features,
    joint_population_apply_feature_weights,
    joint_population_apply_pairwise_weights,
    joint_population_build_pairwise_weights,
    joint_population_default_feature_weights,
)
from .io_utils import (
    discover_population_files,
    extract_group_id_from_filename,
    infer_batch_label,
    load_pred_from_mat,
    parse_formalinsession_group_and_vid,
    resolve_data_path,
    save_json,
    save_mat,
    save_pickle,
)
from .mnn import apply_guided_mnn_correction, fit_guided_mnn_correction
from .motionmapper import (
    build_density_watershed,
    find_templates_from_data,
    find_tdistributed_projections_fmin,
    find_wavelets,
    find_watershed_regions_v2,
    run_tsne_with_config,
)
from .pose import (
    align_pose_sequence,
    cap_pose_frames_for_pca,
    compute_common_alignment_axes,
    joint_population_joint_names,
    joint_population_mnn_guidance_ids,
    joint_population_orientation_qc,
    joint_population_pain_label,
    joint_population_pose_joint_sets,
    joint_population_rigid_segment_qc,
    joint_population_segment_qc,
    joint_population_supervision_ids,
    preprocess_pose_data,
)
from .utils import sanitize_nonfinite_matrix, set_random_seed





def maybe_augment(X: np.ndarray, group_ids: np.ndarray, phase_ids: np.ndarray, supervision: SupervisionConfig, num_groups: int) -> np.ndarray:
    if not supervision.enable:
        return np.asarray(X, dtype=float)
    mode_lower = str(supervision.mode).lower()
    use_aug = mode_lower in {'feature-aug', 'both'}
    if not use_aug:
        return np.asarray(X, dtype=float)
    X = np.asarray(X, dtype=float)
    group_ids = np.asarray(group_ids, dtype=int).reshape(-1)
    phase_ids = np.asarray(phase_ids, dtype=int).reshape(-1)
    n = X.shape[0]
    G = np.zeros((n, int(num_groups)), dtype=float)
    for ii, g in enumerate(group_ids):
        if 1 <= int(g) <= int(num_groups):
            G[ii, int(g) - 1] = 1.0
    num_phases = len(supervision.phase_windows)
    P = np.zeros((n, int(num_phases)), dtype=float)
    for ii, p in enumerate(phase_ids):
        if 1 <= int(p) <= int(num_phases):
            P[ii, int(p) - 1] = 1.0
    return np.concatenate(
        [
            X,
            float(supervision.feature_aug_group_weight) * G,
            float(supervision.feature_aug_phase_weight) * P,
        ],
        axis=1,
    )



def fit_fisher_lda_balanced(X: np.ndarray, y: np.ndarray, n_comp: int, balance_mode: str = 'balanced') -> dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    K = classes.size
    n_comp = int(min(n_comp, K - 1, X.shape[1]))
    if n_comp <= 0:
        return {'mu': np.nanmean(X, axis=0), 'W': np.zeros((X.shape[1], 0), dtype=float), 'nComp': 0, 'classes': classes}
    mu = np.nanmean(X, axis=0)
    X0 = X - mu.reshape(1, -1)
    m = np.nanmean(X0, axis=0)
    sw = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    sb = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for c in classes:
        idx = y == c
        Xk = X0[idx, :]
        nk = Xk.shape[0]
        if nk == 0:
            continue
        mk = np.nanmean(Xk, axis=0)
        if nk <= 1:
            Ck = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        else:
            Ck = np.cov(Xk, rowvar=False, bias=True)
        if str(balance_mode).lower() == 'balanced':
            wk = 1.0 / K
        else:
            wk = nk / X0.shape[0]
        sw = sw + wk * Ck
        dm = (mk - m).reshape(-1, 1)
        sb = sb + wk * (dm @ dm.T)
    eps_reg = 1e-6 * np.trace(sw) / max(sw.shape[0], 1)
    sw = sw + eps_reg * np.eye(sw.shape[0], dtype=float)
    try:
        M = np.linalg.pinv(sw) @ sb
        vals, vecs = np.linalg.eig(M)
    except Exception:
        vals, vecs = np.linalg.eigh(sb)
    order = np.argsort(-np.real(vals))
    W = np.real(vecs[:, order[:n_comp]])
    if W.size:
        W, _ = np.linalg.qr(W)
    return {'mu': mu, 'W': W, 'nComp': n_comp, 'classes': classes}



def default_skeleton() -> dict[str, Any]:
    joints_idx = np.array(
        [
            [1, 2], [1, 3], [2, 3],
            [1, 4], [4, 5], [5, 6], [6, 7],
            [5, 8], [8, 9], [9, 10],
            [5, 11], [11, 12], [12, 13],
            [6, 14], [14, 15], [15, 16],
            [6, 17], [17, 18], [18, 19],
        ],
        dtype=int,
    )
    skeleton = {
        'joints_idx': joints_idx,
        'color': np.array(
            [
                [1.0, 0.6, 0.2], [1.0, 0.6, 0.2], [1.0, 0.6, 0.2],
                [0.2, 0.635, 0.172], [0.2, 0.635, 0.172], [0.2, 0.635, 0.172], [0.2, 0.635, 0.172],
                [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    }
    return skeleton



def _combine_cells(cell_list: list[np.ndarray], axis: int = 0) -> np.ndarray:
    cell_list = [np.asarray(x) for x in cell_list if x is not None and np.size(x)]
    if not cell_list:
        return np.zeros((0, 0), dtype=float)
    return np.concatenate(cell_list, axis=axis)



def _cov_like_matlab(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X must be 2D')
    if X.shape[0] <= 1:
        return np.zeros((X.shape[1], X.shape[1]), dtype=float)
    return np.cov(X, rowvar=False, bias=False)



def _safe_percentile(x: np.ndarray, q: float, fallback: float = float('nan')) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return fallback
    return float(np.nanpercentile(x, q))



def _characteristic_length(ma1: np.ndarray) -> float:
    if ma1.shape[0] == 0 or ma1.shape[2] < 6:
        return float('nan')
    a = np.asarray(ma1[:, :, 0], dtype=float)
    b = np.asarray(ma1[:, :, 5], dtype=float)
    sj = np.sqrt(np.sum((a - b) ** 2, axis=1))
    sj = sj[np.isfinite(sj) & (sj > 0)]
    if sj.size == 0:
        return float('nan')
    return float(np.nanpercentile(sj, 95))



def _feature_blocks_for_pose(
    ma1: np.ndarray,
    Xi: np.ndarray,
    Yi: np.ndarray,
    idx_mask: np.ndarray,
    pairwise_weight_vector: np.ndarray,
    mus: np.ndarray,
    vecs15: np.ndarray,
    x_idx: np.ndarray,
    floor_idx: np.ndarray,
    parameters: Any,
    characteristic_length: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    p1_dist = compute_smoothed_pairwise_distances(ma1, Xi, Yi, idx_mask)
    allz = np.asarray(ma1[:, 2, floor_idx - 1], dtype=float).reshape(-1)
    floor_val = _safe_percentile(allz, 10, fallback=0.0)
    lz = _characteristic_length(ma1)
    if not np.isfinite(lz) or lz <= 0:
        lz = characteristic_length if characteristic_length is not None else float('nan')
    if not np.isfinite(lz) or lz <= 0:
        lz = 60.0
    scale_val = 90.0 / float(lz)
    p1_dist = p1_dist * scale_val
    p1_dist = joint_population_apply_pairwise_weights(p1_dist, pairwise_weight_vector)
    p1_dist = sanitize_nonfinite_matrix(p1_dist)
    p1 = p1_dist - mus.reshape(1, -1)
    proj = p1 @ vecs15
    data, _ = find_wavelets(proj, vecs15.shape[1], parameters)
    amps = np.sum(data, axis=1)
    data2 = np.log(np.maximum(data, np.finfo(float).tiny))
    data2[data2 < -5] = -5
    data2 = sanitize_nonfinite_matrix(data2)
    jv = compute_joint_velocity_features(ma1, x_idx, scale_val)
    p1z = compute_z_coordinate_features(ma1, x_idx, scale_val, floor_val)
    return data2, p1z, jv, amps, floor_val, scale_val



def _build_output_paths(cfg: PipelineConfig) -> dict[str, Path]:
    run_dir = cfg.run_output_dir
    return {
        'run_dir': run_dir,
        'complete_results_mat': run_dir / f'complete_embedding_results_{cfg.run_tag}.mat',
        'complete_results_pkl': run_dir / f'complete_embedding_results_{cfg.run_tag}.pkl',
        'complete_results_json': run_dir / f'complete_embedding_results_{cfg.run_tag}.json',
        'fixed_parameters_json': run_dir / 'fixed_parameters.json',
        'train_embedding_mat': run_dir / f'train_{cfg.run_tag}.mat',
        'train_embedding_pkl': run_dir / f'train_{cfg.run_tag}.pkl',
        'vecs_mus_mat': run_dir / f'vecsMus_{cfg.run_tag}_training.mat',
        'vecs_mus_pkl': run_dir / f'vecsMus_{cfg.run_tag}_training.pkl',
        'mnn_model_mat': run_dir / f'mnn_model_{cfg.run_tag}.mat',
        'mnn_model_pkl': run_dir / f'mnn_model_{cfg.run_tag}.pkl',
        'training_signal_mat': run_dir / f'trainingSignalData_{cfg.run_tag}.mat',
        'training_signal_pkl': run_dir / f'trainingSignalData_{cfg.run_tag}.pkl',
        'lda_model_mat': run_dir / f'lda_model_{cfg.run_tag}.mat',
        'lda_model_pkl': run_dir / f'lda_model_{cfg.run_tag}.pkl',
        'watershed_mat': run_dir / f'watershed_{cfg.run_tag}.mat',
        'watershed_pkl': run_dir / f'watershed_{cfg.run_tag}.pkl',
    }



def _save_core_artifact(path_mat: Path | None, path_pkl: Path | None, payload: dict[str, Any], save_mat_files: bool, save_pickle_files: bool) -> None:
    if save_mat_files and path_mat is not None:
        save_mat(path_mat, payload)
    if save_pickle_files and path_pkl is not None:
        save_pickle(path_pkl, payload)



def _load_existing_pickle(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return pickle.load(f)



def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    set_random_seed(int(cfg.random_seed))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_output_dir.mkdir(parents=True, exist_ok=True)
    paths = _build_output_paths(cfg)

    if cfg.visualization_only:
        if paths['complete_results_pkl'].exists():
            results = _load_existing_pickle(paths['complete_results_pkl'])
            if cfg.analysis_enabled:
                analysis_outputs = create_analysis_outputs(
                    results,
                    cfg.analysis_outputs_dir,
                    sampling_freq=cfg.parameters.sampling_freq,
                    manual_label_csv=cfg.manual_label_csv,
                )
                results['analysisOutputs'] = {k: str(v) for k, v in analysis_outputs.items()}
                results['manualLabelCsv'] = str(cfg.manual_label_csv) if cfg.manual_label_csv is not None else ''
                save_json(cfg.run_output_dir / 'analysis_outputs_manifest.json', results['analysisOutputs'])
                save_json(paths['complete_results_json'], results)
            return results
        raise FileNotFoundError(f'Visualization-only mode requested but {paths["complete_results_pkl"]} does not exist')

    all_files = discover_population_files(cfg.data_root, cfg.datasets_include)
    if cfg.max_files is not None:
        all_files = all_files[: int(cfg.max_files)]
    if not all_files:
        raise FileNotFoundError(f'No compatible pose MAT files found under {cfg.data_root}')

    training_files = list(all_files)
    reembedding_files = list(all_files)
    metadata = []
    for rel in all_files:
        group, vid_num, dataset_tag = parse_formalinsession_group_and_vid(rel)
        metadata.append({'file': rel, 'group': group or 'UNKNOWN', 'vidNum': vid_num, 'dataset': dataset_tag or 'unknown'})

    joint_sets = joint_population_pose_joint_sets(19)
    alignment_indices = {
        'paws': joint_sets['paws'],
        'front': joint_sets['front'],
        'hind': joint_sets['hind'],
        'left': joint_sets['left'],
        'right': joint_sets['right'],
    }

    training_raw_data: list[np.ndarray] = []
    training_data: list[np.ndarray] = []
    training_labels: list[str] = []
    training_preprocess_info: list[Any] = []
    training_alignment_info: list[Any] = []
    total_training_frames = 0
    for rel_path in training_files:
        file_path = resolve_data_path(rel_path, cfg.data_root)
        pred = load_pred_from_mat(file_path)
        pred_data, preprocess_info = preprocess_pose_data(pred, rel_path)
        training_raw_data.append(pred_data)
        training_labels.append(rel_path)
        training_preprocess_info.append(preprocess_info)
        total_training_frames += int(pred_data.shape[0])

    common_alignment_info = compute_common_alignment_axes(training_raw_data, alignment_indices)
    for pred_data in training_raw_data:
        aligned_data, align_info = align_pose_sequence(pred_data, alignment_indices, common_alignment_info)
        training_data.append(aligned_data)
        training_alignment_info.append(align_info)

    skeleton = default_skeleton()
    joint_names = joint_population_joint_names(19)

    if cfg.scientific.write_orientation_qc:
        orientation_train_csv = cfg.run_output_dir / 'orientation_qc_training.csv'
        joint_population_orientation_qc(training_labels, training_data, training_preprocess_info, training_alignment_info, orientation_train_csv)
    else:
        orientation_train_csv = None
    if cfg.scientific.write_segment_qc:
        segment_train_csv = cfg.run_output_dir / 'segment_length_qc_training.csv'
        joint_population_segment_qc(training_labels, training_data, skeleton['joints_idx'], joint_names, segment_train_csv)
    else:
        segment_train_csv = None
    if cfg.scientific.write_rigid_qc:
        rigid_train_csv = cfg.run_output_dir / 'rigid_segment_qc_training.csv'
        joint_population_rigid_segment_qc(training_labels, training_data, joint_names, rigid_train_csv)
    else:
        rigid_train_csv = None

    x_idx = np.arange(1, 20, dtype=int)
    y_idx = np.arange(1, 20, dtype=int)
    Xi, Yi = np.meshgrid(x_idx, y_idx, indexing='xy')
    Xi = Xi.reshape(-1)
    Yi = Yi.reshape(-1)
    idx_mask = np.flatnonzero(Xi != Yi)
    pairwise_weight_vector = joint_population_build_pairwise_weights(Xi, Yi, idx_mask, joint_sets, cfg.scientific)

    lengtht = np.zeros((len(training_data),), dtype=float)
    pca_idx_per_file: list[np.ndarray] = []
    for j, ma1 in enumerate(training_data):
        ma_cap, pca_frame_idx = cap_pose_frames_for_pca(ma1, cfg.pca_pre_frame_cap)
        pca_idx_per_file.append(pca_frame_idx)
        lengtht[j] = _characteristic_length(ma_cap)
    valid_len = lengtht[np.isfinite(lengtht) & (lengtht > 0)]
    fallback_len = float(np.nanmedian(valid_len)) if valid_len.size else 60.0
    if not np.isfinite(fallback_len) or fallback_len <= 0:
        fallback_len = 60.0
    bad_len_mask = ~np.isfinite(lengtht) | (lengtht <= 0)
    lengtht[bad_len_mask] = fallback_len

    rng = np.random.default_rng(int(cfg.random_seed))
    C = None
    mu_sum = None
    L = 0
    batch_size = 30000
    for j, ma1 in enumerate(training_data):
        ma_cap, _ = cap_pose_frames_for_pca(ma1, cfg.pca_pre_frame_cap)
        p1_dist = compute_smoothed_pairwise_distances(ma_cap, Xi, Yi, idx_mask)
        scale_val_pca = float(lengtht[j]) / 90.0
        p1_dist = p1_dist * scale_val_pca
        p1_dist = joint_population_apply_pairwise_weights(p1_dist, pairwise_weight_vector)
        p1_dist = sanitize_nonfinite_matrix(p1_dist)
        if p1_dist.shape[0] <= batch_size or j == 0:
            X = p1_dist[: min(batch_size, p1_dist.shape[0]), :]
        else:
            choose_idx = rng.choice(p1_dist.shape[0], size=batch_size, replace=False)
            X = p1_dist[choose_idx, :]
        c_batch_size = X.shape[0]
        if c_batch_size == 0:
            continue
        temp_mu = np.sum(X, axis=0)
        cov_x = _cov_like_matlab(X)
        block = cov_x * c_batch_size + np.outer(temp_mu, temp_mu) / c_batch_size
        if C is None:
            C = block
            mu_sum = temp_mu
            L = c_batch_size
        else:
            C = C + block
            mu_sum = mu_sum + temp_mu
            L += c_batch_size
    if C is None or mu_sum is None or L <= 0:
        raise RuntimeError('PCA preparation failed because no training samples were available')
    mus = mu_sum / float(L)
    C = C / float(L) - np.outer(mus, mus)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(-vals)
    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])
    n_pca = int(cfg.parameters.pca_modes)
    vecs15 = vecs[:, :n_pca]
    pca_payload = {'C': C, 'L': np.array([[L]], dtype=float), 'mus': mus, 'vals': vals, 'vecs': vecs}
    _save_core_artifact(paths['vecs_mus_mat'], paths['vecs_mus_pkl'], pca_payload, cfg.save_mat_files, cfg.save_pickle_files)

    feature_descriptor = 'Joint cap+formalin embedding (pure Python)'
    if cfg.scientific.enabled:
        feature_descriptor = 'Scientific anchor-guided pain embedding (pure Python)'
    supervision = cfg.supervision
    if supervision.enable:
        if cfg.scientific.enabled:
            feature_descriptor = 'Scientific anchor-guided pain-aware embedding (pure Python)'
        else:
            feature_descriptor = 'Pain-aware LDA + feature-aug embedding (pure Python)'

    wavelet_samples_per_file: list[np.ndarray] = []
    z_samples_per_file: list[np.ndarray] = []
    velocity_samples_per_file: list[np.ndarray] = []
    mA_training_samples: list[np.ndarray] = []
    sample_idx_per_file: list[np.ndarray] = []
    group_ids_per_file: list[np.ndarray] = []
    phase_ids_per_file: list[np.ndarray] = []
    class_ids_per_file: list[np.ndarray] = []
    state_ids_per_file: list[np.ndarray] = []
    anchor_mask_per_file: list[np.ndarray] = []
    pain_binary_per_file: list[np.ndarray] = []
    group_name_per_file: list[np.ndarray] = []
    time_min_per_file: list[np.ndarray] = []
    file_id_per_file: list[np.ndarray] = []
    batch_id_per_file: list[np.ndarray] = []

    feature_weight_spec = joint_population_default_feature_weights(len(x_idx))
    feature_weight_report: dict[str, Any] = {'enabled': False, 'note': 'Using default feature weights'}
    num_phase_windows = len(supervision.phase_windows)
    Fs_embed = float(cfg.parameters.sampling_freq)

    for j, ma1 in enumerate(training_data):
        this_label = training_labels[j]
        grp_str, _ = extract_group_id_from_filename(this_label)
        total_duration_min = ((ma1.shape[0] - 1) / Fs_embed) / 60.0 if ma1.shape[0] else 0.0
        wavelet_block, z_block, velocity_block, amps, _, _ = _feature_blocks_for_pose(
            ma1,
            Xi,
            Yi,
            idx_mask,
            pairwise_weight_vector,
            mus,
            vecs15,
            x_idx,
            joint_sets['floor'],
            cfg.parameters,
            float(lengtht[j]),
        )
        sample_idx = np.arange(0, wavelet_block.shape[0], int(cfg.sample_stride), dtype=int)
        sample_idx = np.unique(sample_idx[(sample_idx >= 0) & (sample_idx < wavelet_block.shape[0])])
        if sample_idx.size == 0 and wavelet_block.shape[0] > 0:
            sample_idx = np.array([0], dtype=int)
        wavelet_samples_per_file.append(wavelet_block[sample_idx, :] if sample_idx.size else np.zeros((0, wavelet_block.shape[1]), dtype=float))
        z_samples_per_file.append(z_block[sample_idx, :] if sample_idx.size else np.zeros((0, z_block.shape[1]), dtype=float))
        velocity_samples_per_file.append(velocity_block[sample_idx, :] if sample_idx.size else np.zeros((0, velocity_block.shape[1]), dtype=float))
        mA_training_samples.append(amps[sample_idx] if sample_idx.size else np.zeros((0,), dtype=float))

        t_samples_min = sample_idx.astype(float) / (Fs_embed * 60.0)
        state_ids_sample, anchor_mask_sample, pain_binary_sample, _ = joint_population_mnn_guidance_ids(
            grp_str, t_samples_min, total_duration_min, cfg.scientific
        )
        sample_idx_per_file.append(sample_idx)
        state_ids_per_file.append(state_ids_sample)
        anchor_mask_per_file.append(anchor_mask_sample)
        pain_binary_per_file.append(pain_binary_sample)
        group_name_per_file.append(np.array([grp_str] * sample_idx.size, dtype=object))
        time_min_per_file.append(t_samples_min)
        file_id_per_file.append(np.full((sample_idx.size,), j + 1, dtype=int))
        batch_id_per_file.append(np.full((sample_idx.size,), j + 1, dtype=int))

        if supervision.enable:
            current_group_ids, phase_ids, class_ids = joint_population_supervision_ids(grp_str, t_samples_min, supervision)
        else:
            current_group_ids = np.ones((sample_idx.size,), dtype=int)
            phase_ids = np.ones((sample_idx.size,), dtype=int)
            class_ids = np.ones((sample_idx.size,), dtype=int)
        group_ids_per_file.append(current_group_ids)
        phase_ids_per_file.append(phase_ids)
        class_ids_per_file.append(class_ids)

    sample_blocks_all = {
        'wavelet': _combine_cells(wavelet_samples_per_file, axis=0),
        'z': _combine_cells(z_samples_per_file, axis=0),
        'velocity': _combine_cells(velocity_samples_per_file, axis=0),
    }
    sample_meta_all = {
        'groupNames': np.concatenate(group_name_per_file, axis=0) if group_name_per_file else np.array([], dtype=object),
        'timeMinutes': np.concatenate(time_min_per_file, axis=0) if time_min_per_file else np.zeros((0,), dtype=float),
        'painBinary': np.concatenate(pain_binary_per_file, axis=0) if pain_binary_per_file else np.zeros((0,), dtype=float),
        'stateIDs': np.concatenate(state_ids_per_file, axis=0) if state_ids_per_file else np.zeros((0,), dtype=int),
        'anchorMask': np.concatenate(anchor_mask_per_file, axis=0) if anchor_mask_per_file else np.zeros((0,), dtype=bool),
        'fileIds': np.concatenate(file_id_per_file, axis=0) if file_id_per_file else np.zeros((0,), dtype=int),
        'batchIds': np.concatenate(batch_id_per_file, axis=0) if batch_id_per_file else np.zeros((0,), dtype=int),
    }
    feature_weight_report = {
        'mode': 'fixed',
        'note': 'Using built-in fixed feature weights',
        'featureWeightLabel': feature_weight_spec.label,
    }
    selected_mnn_k = int(cfg.selected_mnn_k)
    selected_tsne_perplexity = cfg.selected_tsne_perplexity
    fixed_parameters = {
        'mode': 'fixed',
        'featureWeightLabel': feature_weight_spec.label,
        'selectedMnnK': selected_mnn_k,
        'selectedTsnePerplexity': selected_tsne_perplexity,
        'tsneBackend': cfg.tsne_backend,
        'sampleStride': int(cfg.sample_stride),
        'numPerDataset': int(cfg.num_per_dataset),
    }
    feature_descriptor = (
        f'{feature_descriptor} | feature-weights={feature_weight_spec.label} '
        f'| mnnk={selected_mnn_k} | perp={selected_tsne_perplexity}'
    )
    save_json(paths['fixed_parameters_json'], fixed_parameters)

    mD_training_samples: list[np.ndarray] = []
    for j in range(len(training_data)):
        Xw, _ = joint_population_apply_feature_weights(
            wavelet_samples_per_file[j], z_samples_per_file[j], velocity_samples_per_file[j], feature_weight_spec
        )
        mD_training_samples.append(Xw)

    allD_train_raw = _combine_cells(mD_training_samples, axis=0)
    allA_train = np.concatenate(mA_training_samples, axis=0) if mA_training_samples else np.zeros((0,), dtype=float)

    batch_labels = []
    for j, Xw in enumerate(mD_training_samples):
        batch_label_j = infer_batch_label(training_labels[j])
        batch_labels.extend([batch_label_j] * Xw.shape[0])
    if batch_labels:
        _, batch_vector = np.unique(np.asarray(batch_labels, dtype=object), return_inverse=True)
        batch_vector = batch_vector.astype(int) + 1
    else:
        batch_vector = np.zeros((0,), dtype=int)

    if supervision.enable:
        y_class_all = np.concatenate(class_ids_per_file, axis=0).astype(int)
        y_group_all = np.concatenate(group_ids_per_file, axis=0).astype(int)
        y_phase_all = np.concatenate(phase_ids_per_file, axis=0).astype(int)
    else:
        n_total_samples = allD_train_raw.shape[0]
        y_class_all = np.ones((n_total_samples,), dtype=int)
        y_group_all = np.ones((n_total_samples,), dtype=int)
        y_phase_all = np.ones((n_total_samples,), dtype=int)
    state_all = np.concatenate(state_ids_per_file, axis=0).astype(int) if state_ids_per_file else np.zeros((0,), dtype=int)
    anchor_all = np.concatenate(anchor_mask_per_file, axis=0).astype(bool) if anchor_mask_per_file else np.zeros((0,), dtype=bool)
    pain_binary_all = np.concatenate(pain_binary_per_file, axis=0).astype(float) if pain_binary_per_file else np.zeros((0,), dtype=float)

    valid_lab = np.isfinite(y_class_all) & np.all(np.isfinite(allD_train_raw), axis=1)
    X = sanitize_nonfinite_matrix(allD_train_raw[valid_lab, :])
    allA_train = np.asarray(allA_train, dtype=float).reshape(-1)[valid_lab]
    batch_vector = batch_vector[valid_lab]
    y_class_all = y_class_all[valid_lab]
    y_group_all = y_group_all[valid_lab]
    y_phase_all = y_phase_all[valid_lab]
    state_all = state_all[valid_lab]
    anchor_all = anchor_all[valid_lab]
    pain_binary_all = pain_binary_all[valid_lab]

    if cfg.use_mnn_correction:
        mnn_opts = {
            'k': selected_mnn_k,
            'ndim': min(50, X.shape[1]),
            'sigma': None,
            'distance': 'euclidean',
            'verbose': False,
            'stateAware': bool(cfg.scientific.enable_mnn_state_aware),
            'preferAnchorReference': bool(cfg.scientific.prefer_anchor_reference),
            'minStatePoints': int(cfg.scientific.min_state_points),
        }
        mnn_model, allD_train_corrected = fit_guided_mnn_correction(X, batch_vector, mnn_opts, state_all, anchor_all)
        mnn_model['guidanceConfig'] = cfg.scientific
        mnn_model['trainingPainBinary'] = pain_binary_all
        mnn_model['enabled'] = True
        mnn_model['note'] = 'Guided MNN enabled'
    else:
        allD_train_corrected = X
        mnn_model = {
            'enabled': False,
            'mode': 'none',
            'note': 'MNN disabled for pure-Python formalin run',
            'guidanceConfig': cfg.scientific,
        }
    _save_core_artifact(paths['mnn_model_mat'], paths['mnn_model_pkl'], {'mnnModel': mnn_model}, cfg.save_mat_files, cfg.save_pickle_files)

    use_lda = supervision.enable and str(supervision.mode).lower() in {'lda', 'both'}
    use_aug = supervision.enable and str(supervision.mode).lower() in {'feature-aug', 'both'}
    X_train_core = allD_train_corrected
    if use_lda:
        max_comp = min(int(supervision.n_lda), X_train_core.shape[1], np.unique(y_class_all).size - 1)
        lda_model = fit_fisher_lda_balanced(X_train_core, y_class_all, max_comp, supervision.class_balance)
        X_embed = (X_train_core - lda_model['mu'].reshape(1, -1)) @ lda_model['W'] if lda_model['W'].size else X_train_core.copy()
    else:
        lda_model = None
        X_embed = X_train_core
    if use_aug:
        X_embed = maybe_augment(X_embed, y_group_all, y_phase_all, supervision, len(supervision.groups))

    y_data = run_tsne_with_config(X_embed, selected_tsne_perplexity, seed=int(cfg.random_seed), backend=cfg.tsne_backend)
    signal_data, signal_amps = find_templates_from_data(
        X_embed,
        y_data,
        allA_train,
        int(cfg.num_per_dataset),
        cfg.parameters,
        seed=int(cfg.random_seed),
    )
    allD_training = signal_data
    Y_training = run_tsne_with_config(allD_training, selected_tsne_perplexity, seed=int(cfg.random_seed), backend=cfg.tsne_backend)
    watershed_bundle = build_density_watershed(Y_training, sigma_density=0.8, grid_size=501)
    mD_training = [signal_data]
    mA_training = [signal_amps]
    _save_core_artifact(
        paths['training_signal_mat'],
        paths['training_signal_pkl'],
        {'mA_training': mA_training, 'mD_training': mD_training},
        cfg.save_mat_files,
        cfg.save_pickle_files,
    )
    if lda_model is not None:
        _save_core_artifact(paths['lda_model_mat'], paths['lda_model_pkl'], {'ldaModel': lda_model, 'supervision': supervision}, cfg.save_mat_files, cfg.save_pickle_files)

    _save_core_artifact(
        paths['train_embedding_mat'],
        paths['train_embedding_pkl'],
        {'Y_training': Y_training, 'allD_training': allD_training},
        cfg.save_mat_files,
        cfg.save_pickle_files,
    )
    watershed_payload = {
        'D': watershed_bundle['D'],
        'LL': watershed_bundle['LL'],
        'LL2': watershed_bundle['LL2'],
        'llbwb': watershed_bundle['llbwb'],
        'boundaryPolys': watershed_bundle['boundaryPolys'],
        'xx': watershed_bundle['xx'],
    }
    _save_core_artifact(paths['watershed_mat'], paths['watershed_pkl'], watershed_payload, cfg.save_mat_files, cfg.save_pickle_files)

    reembedding_data = list(training_data)
    reembedding_labels = list(training_labels)
    reembedding_metadata = list(metadata)
    reembedding_preprocess_info = list(training_preprocess_info)
    reembedding_alignment_info = list(training_alignment_info)

    if cfg.scientific.write_orientation_qc:
        orientation_reembed_csv = cfg.run_output_dir / 'orientation_qc_reembedding.csv'
        joint_population_orientation_qc(reembedding_labels, reembedding_data, reembedding_preprocess_info, reembedding_alignment_info, orientation_reembed_csv)
    else:
        orientation_reembed_csv = None
    if cfg.scientific.write_segment_qc:
        segment_reembed_csv = cfg.run_output_dir / 'segment_length_qc_reembedding.csv'
        joint_population_segment_qc(reembedding_labels, reembedding_data, skeleton['joints_idx'], joint_names, segment_reembed_csv)
    else:
        segment_reembed_csv = None
    if cfg.scientific.write_rigid_qc:
        rigid_reembed_csv = cfg.run_output_dir / 'rigid_segment_qc_reembedding.csv'
        joint_population_rigid_segment_qc(reembedding_labels, reembedding_data, joint_names, rigid_reembed_csv)
    else:
        rigid_reembed_csv = None

    z_embeddings_all: list[np.ndarray] = []
    wr_fine_all: list[np.ndarray] = []
    features_all: list[np.ndarray] = []
    amps_all: list[np.ndarray] = []
    for j, ma1 in enumerate(reembedding_data):
        wavelet_block, z_block, velocity_block, amps, _, _ = _feature_blocks_for_pose(
            ma1,
            Xi,
            Yi,
            idx_mask,
            pairwise_weight_vector,
            mus,
            vecs15,
            x_idx,
            joint_sets['floor'],
            cfg.parameters,
            float(np.nanmedian(lengtht)),
        )
        nn_data, _ = joint_population_apply_feature_weights(wavelet_block, z_block, velocity_block, feature_weight_spec)
        nn_data = sanitize_nonfinite_matrix(nn_data)
        grp_str, _ = extract_group_id_from_filename(reembedding_labels[j])
        t_min_full = np.arange(nn_data.shape[0], dtype=float) / (Fs_embed * 60.0)
        total_duration_min = ((nn_data.shape[0] - 1) / Fs_embed) / 60.0 if nn_data.shape[0] else 0.0
        state_vec_full, anchor_mask_full, _, _ = joint_population_mnn_guidance_ids(grp_str, t_min_full, total_duration_min, cfg.scientific)
        nn_data_corr = apply_guided_mnn_correction(nn_data, mnn_model, state_vec_full, anchor_mask_full)
        nn_data_embed = nn_data_corr
        if use_lda and lda_model is not None and lda_model['W'].size:
            nn_data_embed = (nn_data_corr - lda_model['mu'].reshape(1, -1)) @ lda_model['W']
        if supervision.enable and use_aug:
            group_vec, phase_ids_full, _ = joint_population_supervision_ids(grp_str, t_min_full, supervision)
            nn_data_embed = maybe_augment(nn_data_embed, group_vec, phase_ids_full, supervision, len(supervision.groups))
        z_values, z_costs, z_guesses, in_conv_hull, mean_max, exit_flags = find_tdistributed_projections_fmin(
            nn_data_embed, allD_training, Y_training, selected_tsne_perplexity, cfg.parameters
        )
        z = np.asarray(z_values, dtype=float)
        z[~in_conv_hull, :] = z_guesses[~in_conv_hull, :]
        wr, _, _, _, _, _ = find_watershed_regions_v2(
            z,
            np.asarray(watershed_bundle['xx'], dtype=float),
            np.asarray(watershed_bundle['LL'], dtype=int),
            0.5,
            1,
            None,
            5,
            None,
            False,
        )
        z_embeddings_all.append(z)
        wr_fine_all.append(wr)
        features_all.append(nn_data_embed)
        amps_all.append(amps)

    results = {
        'training_files': training_files,
        'reembedding_files': reembedding_files,
        'reembedding_labels_all': reembedding_labels,
        'reembedding_metadata_all': reembedding_metadata,
        'zEmbeddings_all': z_embeddings_all,
        'wrFINE_all': wr_fine_all,
        'Y_training': Y_training,
        'D': watershed_bundle['D'],
        'LL': watershed_bundle['LL'],
        'LL2': watershed_bundle['LL2'],
        'llbwb': watershed_bundle['llbwb'],
        'boundaryPolys': watershed_bundle['boundaryPolys'],
        'xx': watershed_bundle['xx'],
        'parameters': cfg.parameters,
        'nPCA': n_pca,
        'skeleton': skeleton,
        'featureDescriptor': feature_descriptor,
        'training_preprocess_info': training_preprocess_info,
        'reembedding_preprocess_info': reembedding_preprocess_info,
        'training_alignment_info': training_alignment_info,
        'reembedding_alignment_info': reembedding_alignment_info,
        'common_alignment_info': common_alignment_info,
        'alignment_mode': 'common_global',
        'features_all': features_all,
        'amps_all': amps_all,
        'supervision': supervision,
        'scientificCfg': cfg.scientific,
        'featureWeightSpec': feature_weight_spec,
        'fixedParameters': fixed_parameters,
        'workflow': 'fixed-parameter -> manual labeling -> graph generation',
        'selectedMnnK': selected_mnn_k,
        'selectedTsnePerplexity': selected_tsne_perplexity,
        'pairwiseWeightVector': pairwise_weight_vector,
        'mnnModel': mnn_model,
        'ldaModel': lda_model,
        'metadata': metadata,
        'totalTrainingFrames': total_training_frames,
        'runTag': cfg.run_tag,
        'dataRoot': str(cfg.data_root),
        'manualLabelCsv': str(cfg.manual_label_csv) if cfg.manual_label_csv is not None else '',
        'embeddingBackend': cfg.tsne_backend,
        'unsupportedSections': [
            'Section 12B region-12 frame classifier',
            'Section 16 semi-supervised re-embedding branch',
            'Section 17 pain-vs-non-pain diagnostics',
        ],
    }
    if orientation_train_csv is not None:
        results['orientationTrainCsv'] = str(orientation_train_csv)
    if orientation_reembed_csv is not None:
        results['orientationReembedCsv'] = str(orientation_reembed_csv)
    if segment_train_csv is not None:
        results['segmentTrainCsv'] = str(segment_train_csv)
    if segment_reembed_csv is not None:
        results['segmentReembedCsv'] = str(segment_reembed_csv)
    if rigid_train_csv is not None:
        results['rigidTrainCsv'] = str(rigid_train_csv)
    if rigid_reembed_csv is not None:
        results['rigidReembedCsv'] = str(rigid_reembed_csv)

    if cfg.save_mat_files:
        save_mat(paths['complete_results_mat'], {'results': results})
    if cfg.save_pickle_files:
        save_pickle(paths['complete_results_pkl'], results)
    if cfg.analysis_enabled:
        analysis_outputs = create_analysis_outputs(
            results,
            cfg.analysis_outputs_dir,
            sampling_freq=cfg.parameters.sampling_freq,
            manual_label_csv=cfg.manual_label_csv,
        )
        results['analysisOutputs'] = {k: str(v) for k, v in analysis_outputs.items()}
        save_json(cfg.run_output_dir / 'analysis_outputs_manifest.json', results['analysisOutputs'])

    save_json(paths['complete_results_json'], results)
    return results
