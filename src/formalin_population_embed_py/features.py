from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .pose import joint_population_pose_joint_sets
from .utils import medfilt1, sanitize_nonfinite_matrix, smooth_moving_average


@dataclass
class FeatureWeightSpec:
    wavelet: float = 1.0
    z: float = 0.25
    velocity: float = 0.5
    z_joint_weights: np.ndarray | None = None
    velocity_joint_weights: np.ndarray | None = None
    hind_gain: float = 1.0
    fore_gain: float = 1.0
    face_weight: float = 1.0
    neck_weight: float = 1.0
    core_weight: float = 1.0
    fore_left_prox_weight: float = 1.0
    fore_right_prox_weight: float = 1.0
    fore_left_distal_weight: float = 1.0
    fore_right_distal_weight: float = 1.0
    hind_left_prox_weight: float = 1.0
    hind_right_prox_weight: float = 1.0
    hind_left_distal_weight: float = 1.0
    hind_right_distal_weight: float = 1.0
    symmetry_penalty: float = 0.0
    regularization_penalty: float = 0.0
    label: str = "fixed_default"

    def __post_init__(self) -> None:
        if self.z_joint_weights is None:
            self.z_joint_weights = np.ones(19, dtype=float)
        else:
            self.z_joint_weights = np.asarray(self.z_joint_weights, dtype=float).reshape(-1)
        if self.velocity_joint_weights is None:
            self.velocity_joint_weights = np.ones(19, dtype=float)
        else:
            self.velocity_joint_weights = np.asarray(self.velocity_joint_weights, dtype=float).reshape(-1)


def joint_population_default_feature_weights(num_joints: int = 19) -> FeatureWeightSpec:
    return FeatureWeightSpec(
        z_joint_weights=np.ones(num_joints, dtype=float),
        velocity_joint_weights=np.ones(num_joints, dtype=float),
        label="fixed_default",
    )


def joint_population_apply_joint_group_weights(
    spec: FeatureWeightSpec,
    joint_sets: dict[str, np.ndarray],
    weights: dict[str, float],
) -> FeatureWeightSpec:
    spec = replace(spec)
    z_weights = np.ones_like(spec.z_joint_weights, dtype=float)
    velocity_weights = np.ones_like(spec.velocity_joint_weights, dtype=float)
    group_map = [
        ("face_weight", "face"),
        ("neck_weight", "neck"),
        ("core_weight", "core"),
        ("fore_left_prox_weight", "foreLeftProx"),
        ("fore_right_prox_weight", "foreRightProx"),
        ("fore_left_distal_weight", "foreLeftDistal"),
        ("fore_right_distal_weight", "foreRightDistal"),
        ("hind_left_prox_weight", "hindLeftProx"),
        ("hind_right_prox_weight", "hindRightProx"),
        ("hind_left_distal_weight", "hindLeftDistal"),
        ("hind_right_distal_weight", "hindRightDistal"),
    ]
    for field_name, joint_field in group_map:
        if field_name not in weights or joint_field not in joint_sets:
            continue
        idx = joint_sets[joint_field].astype(int) - 1
        gain = float(weights[field_name])
        if idx.size == 0 or not np.isfinite(gain):
            continue
        z_weights[idx] *= gain
        velocity_weights[idx] *= gain
        setattr(spec, field_name, gain)
    spec.z_joint_weights = spec.z_joint_weights * z_weights
    spec.velocity_joint_weights = spec.velocity_joint_weights * velocity_weights
    left_fields = ["fore_left_prox_weight", "fore_left_distal_weight", "hind_left_prox_weight", "hind_left_distal_weight"]
    right_fields = ["fore_right_prox_weight", "fore_right_distal_weight", "hind_right_prox_weight", "hind_right_distal_weight"]
    diffs = []
    for left_field, right_field in zip(left_fields, right_fields):
        left_value = max(float(weights.get(left_field, 1.0)), 1e-6)
        right_value = max(float(weights.get(right_field, 1.0)), 1e-6)
        diffs.append(abs(np.log(left_value) - np.log(right_value)))
    spec.symmetry_penalty = float(np.nanmean(diffs)) if diffs else 0.0
    vals = [abs(np.log(max(float(weights.get(field_name, 1.0)), 1e-6))) for field_name, _ in group_map]
    spec.regularization_penalty = float(np.nanmean(vals)) if vals else 0.0
    return spec


def joint_population_apply_feature_weights(
    wavelet_block: np.ndarray | None,
    z_block: np.ndarray | None,
    velocity_block: np.ndarray | None,
    spec: FeatureWeightSpec | None = None,
) -> tuple[np.ndarray, FeatureWeightSpec]:
    if spec is None:
        n_joints = 19
        if z_block is not None and np.size(z_block):
            n_joints = np.asarray(z_block).shape[1]
        elif velocity_block is not None and np.size(velocity_block):
            n_joints = np.asarray(velocity_block).shape[1]
        spec = joint_population_default_feature_weights(n_joints)
    if wavelet_block is None:
        wavelet_block = np.zeros((0, 0), dtype=float)
    if z_block is None:
        z_block = np.zeros((wavelet_block.shape[0], spec.z_joint_weights.size), dtype=float)
    if velocity_block is None:
        velocity_block = np.zeros((wavelet_block.shape[0], spec.velocity_joint_weights.size), dtype=float)
    wavelet_block = sanitize_nonfinite_matrix(np.asarray(wavelet_block, dtype=float))
    z_block = sanitize_nonfinite_matrix(np.asarray(z_block, dtype=float))
    velocity_block = sanitize_nonfinite_matrix(np.asarray(velocity_block, dtype=float))
    z_weighted = spec.z * (z_block * spec.z_joint_weights.reshape(1, -1)) if z_block.size else z_block
    velocity_weighted = spec.velocity * (velocity_block * spec.velocity_joint_weights.reshape(1, -1)) if velocity_block.size else velocity_block
    wavelet_weighted = spec.wavelet * wavelet_block if wavelet_block.size else wavelet_block
    X = np.concatenate([wavelet_weighted, z_weighted, velocity_weighted], axis=1)
    X = sanitize_nonfinite_matrix(X)
    return X, spec


def joint_population_build_pairwise_weights(
    Xi: np.ndarray,
    Yi: np.ndarray,
    idx_mask: np.ndarray,
    joint_sets: dict[str, np.ndarray] | None,
    cfg: Any | None,
) -> np.ndarray:
    joint_sets = joint_sets or joint_population_pose_joint_sets(int(max(np.max(Xi), np.max(Yi))))
    Xi_use = Xi[idx_mask]
    Yi_use = Yi[idx_mask]
    pairwise_weight_vector = np.ones(Xi_use.size, dtype=float)
    if cfg is None or not getattr(cfg.fixed_pairwise_prior, "enable", False):
        return pairwise_weight_vector
    snout_idx = int(joint_sets.get("snout", np.array([1]))[0])
    face_idx = joint_sets.get("face", np.array([1, 2, 3], dtype=int))
    ear_left = int(face_idx[min(1, face_idx.size - 1)])
    ear_right = int(face_idx[min(2, face_idx.size - 1)])
    for a, b, gain in [
        (snout_idx, ear_left, cfg.fixed_pairwise_prior.snout_ear_gain),
        (snout_idx, ear_right, cfg.fixed_pairwise_prior.snout_ear_gain),
        (ear_left, ear_right, cfg.fixed_pairwise_prior.ear_ear_gain),
    ]:
        mask = ((Xi_use == a) & (Yi_use == b)) | ((Xi_use == b) & (Yi_use == a))
        pairwise_weight_vector[mask] *= float(gain)
    return pairwise_weight_vector


def joint_population_apply_pairwise_weights(p1_dist: np.ndarray, pairwise_weight_vector: np.ndarray | None) -> np.ndarray:
    if pairwise_weight_vector is None or p1_dist.size == 0:
        return p1_dist
    return np.asarray(p1_dist, dtype=float) * np.asarray(pairwise_weight_vector, dtype=float).reshape(1, -1)


def compute_smoothed_pairwise_distances(ma1: np.ndarray, Xi: np.ndarray, Yi: np.ndarray, idx_mask: np.ndarray) -> np.ndarray:
    Xi0 = Xi.astype(int) - 1
    Yi0 = Yi.astype(int) - 1
    delta = ma1[:, :, Xi0] - ma1[:, :, Yi0]
    pairwise_raw = np.sqrt(np.sum(delta * delta, axis=1)).T
    pairwise_smooth = smooth_moving_average(medfilt1(pairwise_raw, 3), span=3, axis=1)
    return pairwise_smooth[idx_mask, :].T


def compute_joint_velocity_features(ma1: np.ndarray, x_idx: np.ndarray, scale_val: float) -> np.ndarray:
    x0 = x_idx.astype(int) - 1
    diffs = np.diff(ma1[:, :, x0], axis=0)
    speed = np.sqrt(np.sum(diffs * diffs, axis=1))
    speed = medfilt1(speed, 10)
    jv = np.vstack([np.zeros((1, speed.shape[1]), dtype=float), speed])
    jv = jv * float(scale_val)
    jv[jv >= 5] = 5
    return jv


def compute_z_coordinate_features(ma1: np.ndarray, x_idx: np.ndarray, scale_val: float, floor_val: float) -> np.ndarray:
    x0 = x_idx.astype(int) - 1
    z_series = ma1[:, 2, x0]
    z_series = smooth_moving_average(medfilt1(z_series, 3), span=3, axis=0)
    p1z = (z_series - float(floor_val)) * float(scale_val)
    return sanitize_nonfinite_matrix(p1z)
