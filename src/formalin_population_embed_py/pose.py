from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ScientificConfig, SupervisionConfig
from .io_utils import extract_group_id_from_filename, joint_population_path_info
from .utils import nan_percentile, sanitize_nonfinite_matrix


@dataclass
class PosePreprocessInfo:
    file: str
    original_size: tuple[int, ...]
    orientation_label: str = "none"
    orientation_score: float = float("nan")
    format_summary: str = ""
    joints_original: int = 0
    joints_used: int = 0
    joint_policy: str = "none"
    final_size: tuple[int, ...] = (0, 0, 0)
    format_transform: str = "none"


@dataclass
class AlignmentInfo:
    rotation: np.ndarray
    forward_axis: np.ndarray
    lateral_axis: np.ndarray
    normal_axis: np.ndarray
    reference_center: np.ndarray
    reference_center_rotated: np.ndarray
    sample_frames_used: int
    alignment_mode: str
    sign_canonicalization: dict[str, Any]



def joint_population_pose_joint_sets(num_joints: int = 19) -> dict[str, np.ndarray]:
    joint_sets: dict[str, list[int]] = {}
    if num_joints >= 23:
        joint_sets.update(
            {
                "upper": [4, 5, 6],
                "floor": [11, 15, 19, 23],
                "paws": [11, 15, 19, 23],
                "front": [11, 15],
                "hind": [19, 23],
                "left": [11, 19],
                "right": [15, 23],
                "lower": [11, 15, 19, 23],
            }
        )
    elif num_joints >= 19:
        joint_sets.update(
            {
                "snout": [1],
                "upper": [4, 5, 6],
                "floor": [10, 13, 16, 19],
                "paws": [10, 13, 16, 19],
                "front": [10, 13],
                "hind": [16, 19],
                "left": [10, 16],
                "right": [13, 19],
                "lower": [10, 13, 16, 19],
                "face": [1, 2, 3],
                "neck": [4],
                "core": [5, 6, 7],
                "foreLeftProx": [8, 9],
                "foreRightProx": [11, 12],
                "foreLeftDistal": [10],
                "foreRightDistal": [13],
                "hindLeftProx": [14, 15],
                "hindRightProx": [17, 18],
                "hindLeftDistal": [16],
                "hindRightDistal": [19],
                "foreChain": [8, 9, 10, 11, 12, 13],
                "hindChain": [14, 15, 16, 17, 18, 19],
            }
        )
    elif num_joints >= 14:
        joint_sets.update(
            {
                "snout": [1],
                "upper": [4, 5],
                "floor": [8, 10, 12, 14],
                "paws": [8, 10, 12, 14],
                "front": [8, 10],
                "hind": [12, 14],
                "left": [8, 12],
                "right": [10, 14],
                "lower": [8, 10, 12, 14],
                "face": [1, 2, 3],
                "neck": [4],
                "core": [5, 6],
                "foreLeftProx": [7],
                "foreRightProx": [9],
                "foreLeftDistal": [8],
                "foreRightDistal": [10],
                "hindLeftProx": [11],
                "hindRightProx": [13],
                "hindLeftDistal": [12],
                "hindRightDistal": [14],
                "foreChain": [7, 8, 9, 10],
                "hindChain": [11, 12, 13, 14],
            }
        )
    else:
        floor = list(range(max(1, num_joints - 3 + 1), num_joints + 1))
        joint_sets.update(
            {
                "snout": [1],
                "upper": list(range(1, min(3, num_joints) + 1)),
                "floor": floor,
                "paws": floor,
                "front": floor[: min(2, len(floor))],
                "hind": floor[max(0, len(floor) - 2) :],
                "left": floor[:1],
                "right": floor[-1:],
                "lower": floor,
                "face": list(range(1, min(3, num_joints) + 1)),
                "neck": [min(max(4, 1), num_joints)],
                "core": sorted({min(max(4, 1), num_joints), min(max(5, 1), num_joints)}),
                "foreLeftProx": floor[:1],
                "foreRightProx": floor[-1:],
                "foreLeftDistal": floor[:1],
                "foreRightDistal": floor[-1:],
                "hindLeftProx": floor[:1],
                "hindRightProx": floor[-1:],
                "hindLeftDistal": floor[:1],
                "hindRightDistal": floor[-1:],
                "foreChain": floor[: min(2, len(floor))],
                "hindChain": floor[max(0, len(floor) - 2) :],
            }
        )
    out: dict[str, np.ndarray] = {}
    for key, vals in joint_sets.items():
        arr = np.array(sorted(set(v for v in vals if 1 <= v <= num_joints)), dtype=int)
        out[key] = arr
    return out



def joint_population_joint_names(num_joints: int = 19) -> list[str]:
    base = [
        "Snout",
        "EarL",
        "EarR",
        "NeckB",
        "SpineF",
        "SpineM",
        "TailBase",
        "ForeShdL",
        "ForeElbL",
        "ForepawL",
        "ForeShdR",
        "ForeElbR",
        "ForepawR",
        "HindShdL",
        "HindKneeL",
        "HindpawL",
        "HindShdR",
        "HindKneeR",
        "HindpawR",
    ]
    if num_joints <= len(base):
        return base[:num_joints]
    return base + [f"Joint{k}" for k in range(len(base) + 1, num_joints + 1)]



def joint_population_pain_label(group_name: str, time_minutes: float | None = None) -> str:
    if time_minutes is None or not np.isfinite(time_minutes):
        time_minutes = 0.0
    key = str(group_name).strip().upper()
    if key == "FORMALIN":
        return "PAINFUL"
    if key == "CAPSAICIN":
        return "PAINFUL" if time_minutes < 5 else "NONPAINFUL"
    if key in {"CAPSAICIN_PDX", "CAP_NAIVE", "SALINE", "LIDOCAINE", "NAIVE", "VEHICLE", "WT", "NANOSTING", "RAN", "BONE"}:
        return "NONPAINFUL"
    raise ValueError(f"Unknown joint-population group {group_name!r}")



def joint_population_mnn_guidance_ids(
    group_name: str,
    time_minutes: np.ndarray,
    total_duration_minutes: float | None,
    cfg: ScientificConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if total_duration_minutes is None or not np.isfinite(total_duration_minutes):
        total_duration_minutes = float(np.nanmax(time_minutes)) if np.size(time_minutes) else 0.0
        if not np.isfinite(total_duration_minutes):
            total_duration_minutes = 0.0
    time_minutes = np.asarray(time_minutes, dtype=float).reshape(-1)
    n = time_minutes.size
    state_ids = np.ones(n, dtype=int)
    anchor_mask = np.zeros(n, dtype=bool)
    pain_binary = np.zeros(n, dtype=float)
    grp = str(group_name).strip().upper()
    late_start = max(0.0, float(total_duration_minutes) - float(cfg.control_anchor_minutes))
    is_shared_non_pain_group = grp in cfg.shared_non_pain_groups
    is_control_anchor_group = grp in cfg.control_anchor_groups
    is_acute_pain_group = grp in cfg.acute_pain_groups
    if is_shared_non_pain_group:
        if is_control_anchor_group:
            shared_mask = time_minutes >= late_start
            anchor_mask |= shared_mask
        else:
            shared_mask = np.ones(n, dtype=bool)
        state_ids[shared_mask] = 2
    if is_acute_pain_group:
        acute_mask = time_minutes < float(cfg.acute_pain_minutes)
        state_ids[acute_mask] = 3
        if cfg.include_acute_pain_anchor:
            anchor_mask |= acute_mask
    for i, tm in enumerate(time_minutes):
        try:
            pain_binary[i] = float(joint_population_pain_label(grp, float(tm)) == "PAINFUL")
        except Exception:
            pain_binary[i] = 0.0
    info = {
        "groupName": grp,
        "lateStartMinutes": late_start,
        "sharedNonPainGroup": is_shared_non_pain_group,
        "controlAnchorGroup": is_control_anchor_group,
        "acutePainGroup": is_acute_pain_group,
    }
    return state_ids, anchor_mask, pain_binary, info



def joint_population_supervision_ids(group_name: str, time_minutes: np.ndarray, supervision: SupervisionConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_minutes = np.asarray(time_minutes, dtype=float).reshape(-1)
    n = time_minutes.size
    num_phases = len(supervision.phase_windows)
    phase_ids = np.ones(n, dtype=int) * num_phases
    for pw, (t0, t1) in enumerate(supervision.phase_windows, start=1):
        if math.isinf(t1):
            mask = time_minutes >= t0
        else:
            mask = (time_minutes >= t0) & (time_minutes < t1)
        phase_ids[mask] = pw
    try:
        non_pain_id = supervision.groups.index("NONPAINFUL") + 1
        pain_id = supervision.groups.index("PAINFUL") + 1
    except ValueError as exc:
        raise ValueError("supervision.groups must contain NONPAINFUL and PAINFUL") from exc
    group_ids = np.zeros(n, dtype=int)
    for i, tm in enumerate(time_minutes):
        pain_label = joint_population_pain_label(group_name, float(tm))
        group_ids[i] = pain_id if pain_label == "PAINFUL" else non_pain_id
    class_ids = (group_ids - 1) * num_phases + phase_ids
    return group_ids, phase_ids, class_ids



def normalize_pose_format(raw_pred: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {"transform": "none", "notes": "", "originalClass": type(raw_pred).__name__}
    if raw_pred is None or np.size(raw_pred) == 0:
        pose = np.zeros((0, 3, 0), dtype=float)
        info["transform"] = "empty"
        info["outputSize"] = pose.shape
        return pose, info
    pose = np.asarray(raw_pred).squeeze().astype(float)
    if pose.ndim == 2:
        dims = pose.shape
        if dims[1] % 3 == 0:
            joints = dims[1] // 3
            pose = pose.reshape(dims[0], 3, joints)
            info["transform"] = "reshape_flat"
        elif dims[0] % 3 == 0:
            joints = dims[0] // 3
            pose = pose.reshape(3, joints, dims[1]).transpose(2, 0, 1)
            info["transform"] = "permute_flat"
        else:
            raise ValueError(f"Unable to interpret pose data of size {dims}")
    elif pose.ndim == 3:
        dims = pose.shape
        if dims[1] != 3:
            perm_candidates = [
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ]
            matched = False
            for perm in perm_candidates:
                perm_dims = tuple(dims[p] for p in perm)
                if perm_dims[1] == 3:
                    pose = pose.transpose(perm)
                    info["transform"] = f"permute_{perm[0]+1}{perm[1]+1}{perm[2]+1}"
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unable to align coordinate dimension for size {dims}")
    else:
        raise ValueError(f"Unsupported pose dimensionality {pose.ndim} with size {pose.shape}")
    info["outputSize"] = pose.shape
    return pose, info



def select_joint_indices(num_joints: int, region_type: str) -> np.ndarray:
    region_type = region_type.lower()
    if region_type == "upper":
        if num_joints >= 19:
            idx = [4, 5, 6]
        elif num_joints >= 14:
            idx = [4, 5]
        else:
            idx = list(range(1, min(3, num_joints) + 1))
    elif region_type == "lower":
        if num_joints >= 19:
            idx = [10, 13, 16, 19]
        elif num_joints >= 14:
            idx = [8, 10, 12, 14]
        else:
            idx = list(range(max(num_joints - 3 + 1, 1), num_joints + 1))
    else:
        idx = []
    return np.array([i for i in idx if 1 <= i <= num_joints], dtype=int)



def _orientation_transforms() -> list[tuple[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], str]]:
    return [
        (lambda x, y, z: np.stack([x, y, z], axis=1), "[X, Y, Z]"),
        (lambda x, y, z: np.stack([x, -y, -z], axis=1), "[X, -Y, -Z]"),
        (lambda x, y, z: np.stack([x, z, -y], axis=1), "[X, Z, -Y]"),
        (lambda x, y, z: np.stack([x, -z, y], axis=1), "[X, -Z, Y]"),
        (lambda x, y, z: np.stack([y, x, -z], axis=1), "[Y, X, -Z]"),
        (lambda x, y, z: np.stack([y, -x, z], axis=1), "[Y, -X, Z]"),
        (lambda x, y, z: np.stack([z, y, -x], axis=1), "[Z, Y, -X]"),
        (lambda x, y, z: np.stack([z, -y, x], axis=1), "[Z, -Y, X]"),
        (lambda x, y, z: np.stack([-x, y, -z], axis=1), "[-X, Y, -Z]"),
        (lambda x, y, z: np.stack([-x, -y, z], axis=1), "[-X, -Y, Z]"),
        (lambda x, y, z: np.stack([-y, x, z], axis=1), "[-Y, X, Z]"),
        (lambda x, y, z: np.stack([-y, -x, -z], axis=1), "[-Y, -X, -Z]"),
    ]



def apply_orientation_heuristic(pose: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {"applied": False, "bestLabel": "", "score": float("nan"), "scores": [], "warning": ""}
    if pose.size == 0:
        return pose, info
    if pose.shape[1] != 3:
        raise ValueError("Pose data must have size[:,1] == 3 to represent XYZ coordinates")
    num_joints = pose.shape[2]
    idx_upper = select_joint_indices(num_joints, "upper") - 1
    idx_lower = select_joint_indices(num_joints, "lower") - 1
    if idx_upper.size == 0 or idx_lower.size == 0:
        info["warning"] = "Insufficient joints for orientation heuristic"
        return pose, info
    transforms = _orientation_transforms()
    scores = np.full(len(transforms), np.nan, dtype=float)
    x = pose[:, 0, :]
    y = pose[:, 1, :]
    z = pose[:, 2, :]
    for t, (fn, _) in enumerate(transforms):
        candidate = fn(x, y, z)
        z_upper = candidate[:, 2, idx_upper]
        z_lower = candidate[:, 2, idx_lower]
        diff_vals = np.nanmean(z_upper, axis=1) - np.nanmean(z_lower, axis=1)
        scores[t] = float(np.nanmean(diff_vals))
    info["scores"] = scores
    if np.all(~np.isfinite(scores)):
        info["warning"] = "Orientation heuristic returned non-finite scores"
        return pose, info
    best_idx = int(np.nanargmax(scores))
    best_score = float(scores[best_idx])
    info["score"] = best_score
    info["bestLabel"] = transforms[best_idx][1]
    pose = transforms[best_idx][0](x, y, z)
    info["applied"] = True
    if best_score < 0:
        info["warning"] = "Best orientation score is negative"
    return pose, info



def preprocess_pose_data(raw_pred: np.ndarray, file_label: str = "") -> tuple[np.ndarray, PosePreprocessInfo]:
    info = PosePreprocessInfo(file=file_label, original_size=tuple(np.asarray(raw_pred).shape))
    pose, format_info = normalize_pose_format(raw_pred)
    info.format_transform = str(format_info.get("transform", "none"))
    info.joints_original = int(pose.shape[2]) if pose.ndim == 3 else 0
    if pose.shape[2] > 19:
        pose = pose[:, :, :19]
        info.joint_policy = "truncate_first_19"
    elif pose.shape[2] == 19:
        info.joint_policy = "native_19"
    else:
        raise ValueError(f"File {file_label} has {pose.shape[2]} joints; expected >= 19 joints")
    info.joints_used = int(pose.shape[2])
    info.final_size = tuple(pose.shape)
    info.format_summary = f"{pose.shape[0]}x{pose.shape[1]}x{pose.shape[2]}"
    pose, orientation_info = apply_orientation_heuristic(pose)
    info.orientation_label = str(orientation_info.get("bestLabel", "none")) or "none"
    info.orientation_score = float(orientation_info.get("score", float("nan")))
    return pose, info



def _mean_points(frame: np.ndarray, idx_1_based: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx_1_based, dtype=int) - 1
    if idx.size == 0:
        return np.full((3,), np.nan, dtype=float)
    return np.nanmean(frame[:, idx], axis=1)



def compute_alignment_sign_metrics(pose_seq: np.ndarray, idx_struct: dict[str, np.ndarray]) -> tuple[float, float]:
    if pose_seq.size == 0:
        return float("nan"), float("nan")
    frame_count = pose_seq.shape[0]
    sample_count = min(frame_count, 1000)
    sample_idx = np.unique(np.round(np.linspace(0, frame_count - 1, sample_count)).astype(int))
    pose_sample = pose_seq[sample_idx, :, :]
    front_x = np.nanmean(pose_sample[:, 0, idx_struct["front"] - 1], axis=1)
    hind_x = np.nanmean(pose_sample[:, 0, idx_struct["hind"] - 1], axis=1)
    right_y = np.nanmean(pose_sample[:, 1, idx_struct["right"] - 1], axis=1)
    left_y = np.nanmean(pose_sample[:, 1, idx_struct["left"] - 1], axis=1)
    return float(np.nanmean(front_x - hind_x)), float(np.nanmean(right_y - left_y))



def canonicalize_aligned_pose_signs(pose_seq: np.ndarray, idx_struct: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    info = {
        "flipX": False,
        "flipY": False,
        "frontMinusHindXBefore": float("nan"),
        "rightMinusLeftYBefore": float("nan"),
        "frontMinusHindXAfter": float("nan"),
        "rightMinusLeftYAfter": float("nan"),
    }
    if pose_seq.size == 0:
        return pose_seq, info
    front_minus_hind_x, right_minus_left_y = compute_alignment_sign_metrics(pose_seq, idx_struct)
    info["frontMinusHindXBefore"] = front_minus_hind_x
    info["rightMinusLeftYBefore"] = right_minus_left_y
    if np.isfinite(front_minus_hind_x) and front_minus_hind_x < 0:
        pose_seq[:, 0, :] = -pose_seq[:, 0, :]
        info["flipX"] = True
    front_minus_hind_x, right_minus_left_y = compute_alignment_sign_metrics(pose_seq, idx_struct)
    if np.isfinite(right_minus_left_y) and right_minus_left_y < 0:
        pose_seq[:, 1, :] = -pose_seq[:, 1, :]
        info["flipY"] = True
    front_minus_hind_x, right_minus_left_y = compute_alignment_sign_metrics(pose_seq, idx_struct)
    info["frontMinusHindXAfter"] = front_minus_hind_x
    info["rightMinusLeftYAfter"] = right_minus_left_y
    return pose_seq, info



def compute_alignment_axes(pose_seq: np.ndarray, idx_struct: dict[str, np.ndarray]) -> dict[str, Any]:
    num_frames = pose_seq.shape[0]
    sample_count = min(5000, num_frames)
    sample_idx = np.unique(np.round(np.linspace(0, num_frames - 1, sample_count)).astype(int))
    forward_sum = np.zeros(3, dtype=float)
    lateral_sum = np.zeros(3, dtype=float)
    normal_sum = np.zeros(3, dtype=float)
    center_sum = np.zeros(3, dtype=float)
    valid_count = 0
    for idx in sample_idx:
        frame = pose_seq[idx, :, :].T.T  # keep array copy semantics simple
        frame = np.asarray(pose_seq[idx, :, :], dtype=float).T.T
        frame = pose_seq[idx, :, :]
        front_pos = _mean_points(frame, idx_struct["front"])
        hind_pos = _mean_points(frame, idx_struct["hind"])
        left_pos = _mean_points(frame, idx_struct["left"])
        right_pos = _mean_points(frame, idx_struct["right"])
        if not (np.all(np.isfinite(front_pos)) and np.all(np.isfinite(hind_pos)) and np.all(np.isfinite(left_pos)) and np.all(np.isfinite(right_pos))):
            continue
        forward_vec = front_pos - hind_pos
        lateral_vec = right_pos - left_pos
        if np.linalg.norm(forward_vec) < 1e-6 or np.linalg.norm(lateral_vec) < 1e-6:
            continue
        normal_vec = np.cross(forward_vec, lateral_vec)
        if np.linalg.norm(normal_vec) < 1e-6:
            continue
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        if normal_vec[2] < 0:
            normal_vec = -normal_vec
            lateral_vec = -lateral_vec
        forward_proj = forward_vec - np.dot(forward_vec, normal_vec) * normal_vec
        lateral_proj = lateral_vec - np.dot(lateral_vec, normal_vec) * normal_vec
        if np.linalg.norm(forward_proj) < 1e-6 or np.linalg.norm(lateral_proj) < 1e-6:
            continue
        forward_proj = forward_proj / np.linalg.norm(forward_proj)
        lateral_proj = lateral_proj / np.linalg.norm(lateral_proj)
        paw_center = _mean_points(frame, idx_struct["paws"])
        if not np.all(np.isfinite(paw_center)):
            continue
        forward_sum += forward_proj
        lateral_sum += lateral_proj
        normal_sum += normal_vec
        center_sum += paw_center
        valid_count += 1
    if valid_count == 0 or np.linalg.norm(normal_sum) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        center = np.nanmean(np.nanmean(pose_seq, axis=0), axis=1)
        if not np.all(np.isfinite(center)):
            center = np.zeros(3, dtype=float)
    else:
        normal = normal_sum / np.linalg.norm(normal_sum)
        forward_raw = forward_sum - np.dot(forward_sum, normal) * normal
        if np.linalg.norm(forward_raw) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            forward = forward_raw / np.linalg.norm(forward_raw)
        center = center_sum / valid_count
    lateral = np.cross(normal, forward)
    if np.linalg.norm(lateral) < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        lateral = lateral / np.linalg.norm(lateral)
    R = np.column_stack([forward, lateral, normal])
    return {
        "rotation": R,
        "forward": forward,
        "lateral": lateral,
        "normal": normal,
        "referenceCenter": center,
        "sampleCount": valid_count,
    }



def compute_common_alignment_axes(pose_cell: list[np.ndarray], idx_struct: dict[str, np.ndarray]) -> dict[str, Any]:
    forward_sum = np.zeros(3, dtype=float)
    normal_sum = np.zeros(3, dtype=float)
    center_sum = np.zeros(3, dtype=float)
    weight_sum = 0.0
    videos_used = 0
    for pose_seq in pose_cell:
        if pose_seq is None or pose_seq.size == 0:
            continue
        local_axes = compute_alignment_axes(pose_seq, idx_struct)
        if local_axes["sampleCount"] <= 0:
            continue
        if not (np.all(np.isfinite(local_axes["forward"])) and np.all(np.isfinite(local_axes["normal"])) and np.all(np.isfinite(local_axes["referenceCenter"]))):
            continue
        w = max(1.0, float(local_axes["sampleCount"]))
        forward_sum += w * local_axes["forward"]
        normal_sum += w * local_axes["normal"]
        center_sum += w * local_axes["referenceCenter"]
        weight_sum += w
        videos_used += 1
    if weight_sum <= 0 or np.linalg.norm(normal_sum) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        center = np.zeros(3, dtype=float)
        lateral = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        normal = normal_sum / np.linalg.norm(normal_sum)
        forward_raw = forward_sum - np.dot(forward_sum, normal) * normal
        if np.linalg.norm(forward_raw) < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            forward = forward_raw / np.linalg.norm(forward_raw)
        lateral = np.cross(normal, forward)
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            lateral = lateral / np.linalg.norm(lateral)
        center = center_sum / weight_sum
        if not np.all(np.isfinite(center)):
            center = np.zeros(3, dtype=float)
    R = np.column_stack([forward, lateral, normal])
    return {
        "rotation": R,
        "forward": forward,
        "lateral": lateral,
        "normal": normal,
        "referenceCenter": center,
        "sampleCount": int(round(weight_sum)),
        "videosUsed": videos_used,
    }



def align_pose_sequence(
    pose_seq: np.ndarray,
    idx_struct: dict[str, np.ndarray],
    axes_info_override: dict[str, Any] | None = None,
) -> tuple[np.ndarray, AlignmentInfo]:
    if pose_seq.size == 0:
        aligned_seq = pose_seq.copy()
        info = AlignmentInfo(
            rotation=np.eye(3),
            forward_axis=np.array([1.0, 0.0, 0.0]),
            lateral_axis=np.array([0.0, 1.0, 0.0]),
            normal_axis=np.array([0.0, 0.0, 1.0]),
            reference_center=np.zeros(3),
            reference_center_rotated=np.zeros(3),
            sample_frames_used=0,
            alignment_mode="common_global" if axes_info_override else "per_video",
            sign_canonicalization={},
        )
        return aligned_seq, info
    use_override = axes_info_override is not None
    axes_info = axes_info_override if use_override else compute_alignment_axes(pose_seq, idx_struct)
    R = np.asarray(axes_info["rotation"], dtype=float)
    ref_center = np.asarray(axes_info["referenceCenter"], dtype=float).reshape(3, 1)
    aligned_seq = np.zeros_like(pose_seq, dtype=float)
    num_frames = pose_seq.shape[0]
    for f in range(num_frames):
        frame = np.asarray(pose_seq[f, :, :], dtype=float)
        aligned_frame = R.T @ (frame - ref_center)
        aligned_seq[f, :, :] = aligned_frame
    aligned_seq, sign_info = canonicalize_aligned_pose_signs(aligned_seq, idx_struct)
    info = AlignmentInfo(
        rotation=R,
        forward_axis=np.asarray(axes_info["forward"], dtype=float),
        lateral_axis=np.asarray(axes_info["lateral"], dtype=float),
        normal_axis=np.asarray(axes_info["normal"], dtype=float),
        reference_center=np.asarray(axes_info["referenceCenter"], dtype=float),
        reference_center_rotated=np.zeros(3),
        sample_frames_used=int(axes_info.get("sampleCount", 0)),
        alignment_mode="common_global" if use_override else "per_video",
        sign_canonicalization=sign_info,
    )
    return aligned_seq, info



def cap_pose_frames_for_pca(pose_in: np.ndarray, frame_cap: int | float) -> tuple[np.ndarray, np.ndarray]:
    num_frames = pose_in.shape[0]
    if not np.isfinite(frame_cap) or frame_cap <= 0 or num_frames <= frame_cap:
        idx = np.arange(num_frames, dtype=int)
        return pose_in, idx
    idx = np.unique(np.round(np.linspace(0, num_frames - 1, int(frame_cap))).astype(int))
    if idx.size == 0:
        idx = np.array([0], dtype=int)
    idx[-1] = num_frames - 1
    return pose_in[idx, :, :], idx



def joint_population_orientation_qc(
    labels: list[str],
    pose_cell: list[np.ndarray],
    preprocess_info: list[PosePreprocessInfo | None],
    alignment_info: list[AlignmentInfo | None],
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    n = len(labels)
    if n == 0:
        return pd.DataFrame()
    num_joints = 19
    for pose in pose_cell:
        if pose is not None and pose.size > 0:
            num_joints = pose.shape[2]
            break
    joint_sets = joint_population_pose_joint_sets(num_joints)
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        row: dict[str, Any] = {
            "file": label,
            "group": joint_population_path_info(label).get("group", "UNKNOWN"),
            "preprocessOrientation": "",
            "preprocessOrientationScore": float("nan"),
            "headMinusPawZ": float("nan"),
            "frontMinusHindX": float("nan"),
            "rightMinusLeftY": float("nan"),
            "normalZ": float("nan"),
            "forwardX": float("nan"),
            "qcPass": False,
        }
        if i < len(preprocess_info) and preprocess_info[i] is not None:
            row["preprocessOrientation"] = preprocess_info[i].orientation_label
            row["preprocessOrientationScore"] = preprocess_info[i].orientation_score
        if i < len(alignment_info) and alignment_info[i] is not None:
            row["normalZ"] = float(alignment_info[i].normal_axis[2]) if alignment_info[i].normal_axis.size >= 3 else float("nan")
            row["forwardX"] = float(alignment_info[i].forward_axis[0]) if alignment_info[i].forward_axis.size >= 1 else float("nan")
        pose = pose_cell[i]
        if pose is not None and pose.size > 0:
            frame_count = pose.shape[0]
            sample_count = min(frame_count, 1000)
            sample_idx = np.unique(np.round(np.linspace(0, frame_count - 1, sample_count)).astype(int))
            pose_sample = pose[sample_idx, :, :]
            upper_z = np.nanmean(pose_sample[:, 2, joint_sets["upper"] - 1], axis=1)
            paw_z = np.nanmean(pose_sample[:, 2, joint_sets["floor"] - 1], axis=1)
            front_x = np.nanmean(pose_sample[:, 0, joint_sets["front"] - 1], axis=1)
            hind_x = np.nanmean(pose_sample[:, 0, joint_sets["hind"] - 1], axis=1)
            right_y = np.nanmean(pose_sample[:, 1, joint_sets["right"] - 1], axis=1)
            left_y = np.nanmean(pose_sample[:, 1, joint_sets["left"] - 1], axis=1)
            row["headMinusPawZ"] = float(np.nanmean(upper_z - paw_z))
            row["frontMinusHindX"] = float(np.nanmean(front_x - hind_x))
            row["rightMinusLeftY"] = float(np.nanmean(right_y - left_y))
            row["qcPass"] = bool(
                row["headMinusPawZ"] > 0
                and row["frontMinusHindX"] > 0
                and row["rightMinusLeftY"] > 0
                and row["normalZ"] > 0
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df



def joint_population_segment_qc(
    labels: list[str], pose_cells: list[np.ndarray], skeleton_edges: np.ndarray, joint_names: list[str] | None = None, out_csv: str | Path | None = None
) -> pd.DataFrame:
    if not labels:
        labels = [f"file_{k:03d}" for k in range(len(pose_cells))]
    if skeleton_edges is None or np.size(skeleton_edges) == 0:
        return pd.DataFrame()
    if joint_names is None and pose_cells and pose_cells[0] is not None:
        joint_names = joint_population_joint_names(pose_cells[0].shape[2])
    joint_names = joint_names or []
    rows: list[dict[str, Any]] = []
    for file_idx, pose in enumerate(pose_cells):
        if pose is None or pose.size == 0:
            continue
        for edge_idx in range(skeleton_edges.shape[0]):
            a = int(skeleton_edges[edge_idx, 0]) - 1
            b = int(skeleton_edges[edge_idx, 1]) - 1
            if a >= pose.shape[2] or b >= pose.shape[2]:
                continue
            d = np.sqrt(np.sum((pose[:, :, a] - pose[:, :, b]) ** 2, axis=1))
            d = d[np.isfinite(d) & (d > 0)]
            if d.size == 0:
                continue
            rows.append(
                {
                    "file": labels[file_idx],
                    "edgeIndex": edge_idx + 1,
                    "jointA": joint_names[a] if a < len(joint_names) else f"Joint{a+1}",
                    "jointB": joint_names[b] if b < len(joint_names) else f"Joint{b+1}",
                    "medianLength": float(np.nanmedian(d)),
                    "lengthCV": float(np.nanstd(d) / max(np.nanmean(d), np.finfo(float).eps)),
                    "p95minusP5": float(np.nanpercentile(d, 95) - np.nanpercentile(d, 5)),
                }
            )
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df



def joint_population_rigid_segment_qc(
    labels: list[str], pose_cells: list[np.ndarray], joint_names: list[str] | None = None, out_csv: str | Path | None = None
) -> pd.DataFrame:
    if not labels:
        labels = [f"file_{k:03d}" for k in range(len(pose_cells))]
    if joint_names is None and pose_cells and pose_cells[0] is not None:
        joint_names = joint_population_joint_names(pose_cells[0].shape[2])
    joint_names = joint_names or []
    rigid_pairs = np.array([[1, 2], [1, 3], [2, 3]], dtype=int)
    rows: list[dict[str, Any]] = []
    for file_idx, pose in enumerate(pose_cells):
        if pose is None or pose.size == 0:
            continue
        for pair_idx in range(rigid_pairs.shape[0]):
            a = int(rigid_pairs[pair_idx, 0]) - 1
            b = int(rigid_pairs[pair_idx, 1]) - 1
            if a >= pose.shape[2] or b >= pose.shape[2]:
                continue
            d = np.sqrt(np.sum((pose[:, :, a] - pose[:, :, b]) ** 2, axis=1))
            d = d[np.isfinite(d) & (d > 0)]
            if d.size == 0:
                continue
            cv_val = float(np.nanstd(d) / max(np.nanmean(d), np.finfo(float).eps))
            rows.append(
                {
                    "file": labels[file_idx],
                    "pairIndex": pair_idx + 1,
                    "jointA": joint_names[a] if a < len(joint_names) else f"Joint{a+1}",
                    "jointB": joint_names[b] if b < len(joint_names) else f"Joint{b+1}",
                    "medianLength": float(np.nanmedian(d)),
                    "lengthCV": cv_val,
                    "p95minusP5": float(np.nanpercentile(d, 95) - np.nanpercentile(d, 5)),
                    "suspectRigidDrift": bool(cv_val > 0.15),
                }
            )
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df
