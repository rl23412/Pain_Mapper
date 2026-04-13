from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from .config import canonical_dataset_name
from .utils import to_serializable


POPULATION_PREFIX = "/hpc/group/1920-paindetect/population/"


def load_mat_any(path: str | Path) -> dict[str, Any]:
    path = str(path)
    try:
        data = loadmat(path, squeeze_me=False, struct_as_record=False)
        return data
    except NotImplementedError:
        return load_hdf5_mat(path)
    except ValueError:
        return load_hdf5_mat(path)



def _read_hdf5_value(f: h5py.File, obj: Any) -> Any:
    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        return arr
    if isinstance(obj, h5py.Group):
        return {k: _read_hdf5_value(f, v) for k, v in obj.items()}
    if isinstance(obj, h5py.Reference):
        return _read_hdf5_value(f, f[obj])
    return obj



def load_hdf5_mat(path: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            out[key] = _read_hdf5_value(f, f[key])
    return out



def load_pred_from_mat(path: str | Path) -> np.ndarray:
    data = load_mat_any(path)
    if "pred" not in data:
        raise KeyError(f"File {path} does not contain 'pred'")
    pred = data["pred"]
    pred = np.asarray(pred)
    return pred



def save_mat(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, to_serializable(data), do_compression=True)



def save_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)



def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)



def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)



def joint_population_path_info(path_in: str | Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "relPath": "",
        "dataset": "",
        "datasetTag": "",
        "group": "UNKNOWN",
        "vidNum": float("nan"),
        "isCanonicalPose": False,
    }
    if path_in is None:
        return info
    raw_path = str(path_in).strip().replace("\\", "/")
    if not raw_path:
        return info
    match_path = raw_path.lower()
    prefix_idx = match_path.find(POPULATION_PREFIX)
    if prefix_idx >= 0:
        start = prefix_idx + len(POPULATION_PREFIX)
        raw_path = raw_path[start:]
        match_path = match_path[start:]
    raw_path = re.sub(r"^/+", "", raw_path)
    match_path = re.sub(r"^/+", "", match_path)

    cap_tokens = re.match(r"^(?:cap/)?(capsaicin-pdx\d+|capsaicin\d+|naive\d+)/(save_data_avg(?:_[^/]+)?\.mat)$", match_path)
    if cap_tokens:
        token = cap_tokens.group(1)
        file_name = re.sub(r"^save_data_avg", "save_data_AVG", cap_tokens.group(2))
        info.update(
            {
                "dataset": "cap",
                "datasetTag": "cap",
                "relPath": f"cap/{token}/{file_name}",
                "vidNum": parse_numeric_suffix(token),
                "isCanonicalPose": cap_tokens.group(2) == "save_data_avg.mat",
            }
        )
        if token.startswith("capsaicin-pdx"):
            info["group"] = "CAPSAICIN_PDX"
        elif token.startswith("capsaicin"):
            info["group"] = "CAPSAICIN"
        else:
            info["group"] = "NAIVE"
        return info

    formalin_tokens = re.match(
        r"^(?:formalin/)?(saline|formaline|formalin|lidocane|lidocaine)/vid(\d+)/(save_data_avg(?:_[^/]+)?\.mat)$",
        match_path,
    )
    if formalin_tokens:
        token = formalin_tokens.group(1)
        file_name = re.sub(r"^save_data_avg", "save_data_AVG", formalin_tokens.group(3))
        info.update(
            {
                "dataset": "formalin",
                "datasetTag": "formalin",
                "relPath": f"formalin/{token}/vid{formalin_tokens.group(2)}/{file_name}",
                "vidNum": float(formalin_tokens.group(2)),
                "isCanonicalPose": formalin_tokens.group(3) == "save_data_avg.mat",
            }
        )
        if token == "saline":
            info["group"] = "SALINE"
        elif token in {"formalin", "formaline"}:
            info["group"] = "FORMALIN"
        else:
            info["group"] = "LIDOCAINE"
    return info



def parse_numeric_suffix(token: str) -> float:
    m = re.search(r"(\d+)$", token)
    return float(m.group(1)) if m else float("nan")



def parse_formalinsession_group_and_vid(filename: str | Path) -> tuple[str, float, str]:
    info = joint_population_path_info(filename)
    if info.get("dataset"):
        return str(info["group"]), float(info["vidNum"]), str(info["datasetTag"])
    base = Path(str(filename)).name.lower()
    flat_match = re.match(r"(saline|formalin|formaline|lidocaine|lidocane|naive|capsaicin|capsaicin_pdx)_mouse(\d+)\.mat$", base)
    if flat_match:
        token = flat_match.group(1)
        vid_num = float(flat_match.group(2))
        if token == "saline":
            return "SALINE", vid_num, "formalin"
        if token in {"formalin", "formaline"}:
            return "FORMALIN", vid_num, "formalin"
        if token in {"lidocaine", "lidocane"}:
            return "LIDOCAINE", vid_num, "formalin"
        if token == "naive":
            return "NAIVE", vid_num, "bone"
        if token == "capsaicin_pdx":
            return "CAPSAICIN_PDX", vid_num, "cap"
        if token == "capsaicin":
            return "CAPSAICIN", vid_num, "cap"
    rel_path = str(filename).replace("\\", "/").lower()
    parts = [p.strip() for p in rel_path.split("/") if p.strip()]
    group = ""
    dataset_tag = ""
    for token in parts:
        if token == "ran":
            dataset_tag = "ran"
            if not group:
                group = "RAN"
        elif token == "bone":
            dataset_tag = "bone"
        elif token == "saline":
            group = "SALINE"
            dataset_tag = "saline"
            break
        elif token == "wt":
            group = "SALINE"
            dataset_tag = "wt"
            break
        elif token in {"formalin", "formaline"}:
            group = "FORMALIN"
            dataset_tag = token
            break
        elif token == "cpfull":
            group = "FORMALIN"
            dataset_tag = "cpfull"
            break
        elif token in {"lidocaine", "lidocane"}:
            group = "LIDOCAINE"
            dataset_tag = token
            break
        if token.startswith("naive"):
            group = "NAIVE"
            dataset_tag = dataset_tag or "bone"
        elif token.startswith("nano-sting") or token.startswith("nanosting"):
            group = "NANOSTING"
            dataset_tag = dataset_tag or "bone"
        elif token.startswith("vehicle"):
            group = "VEHICLE"
            dataset_tag = dataset_tag or "bone"
    vid_num = float("nan")
    m = re.search(r"vid(\d+)", rel_path)
    if not m:
        m = re.search(r"(?:sample|naive[_-]|nano-sting[_-]|nanosting[_-]|vehicle[_-])(\d+)", rel_path)
    if m:
        vid_num = float(m.group(1))
    return group, vid_num, dataset_tag



def infer_batch_label(filename: str | Path) -> str:
    rel_path = str(filename).replace("\\", "/").lower().strip()
    label = re.sub(r"[^a-z0-9]+", "_", rel_path)
    label = re.sub(r"^_+|_+$", "", label)
    return label or "batch_unknown"



def extract_group_id_from_filename(filename: str | Path) -> tuple[str, float]:
    grp, vid, _ = parse_formalinsession_group_and_vid(filename)
    return (grp.upper() if grp else "UNKNOWN", vid)



def normalize_rel_path(path_in: str | Path, data_root: str | Path, base_dir: str | Path | None = None) -> str:
    rel_path = ""
    if not path_in:
        return rel_path
    norm_path = str(path_in).replace("\\", "/").strip()
    if not norm_path:
        return rel_path
    is_unix_abs = norm_path.startswith("/")
    is_win_abs = re.match(r"^[A-Za-z]:[\\/]", norm_path) is not None
    if base_dir and not (is_unix_abs or is_win_abs):
        norm_path = str(Path(base_dir) / norm_path).replace("\\", "/")
    norm_root = str(data_root).replace("\\", "/")
    helper_info = joint_population_path_info(norm_path)
    if helper_info["dataset"] and helper_info["isCanonicalPose"]:
        return str(helper_info["relPath"])
    root_prefix = norm_root.rstrip("/") + "/"
    if norm_path.startswith(root_prefix):
        candidate_rel = norm_path[len(root_prefix) :]
        helper_info = joint_population_path_info(candidate_rel)
        if helper_info["dataset"] and helper_info["isCanonicalPose"]:
            return str(helper_info["relPath"])
        if (Path(data_root) / candidate_rel).is_file():
            return candidate_rel
    if (Path(data_root) / norm_path).is_file():
        helper_info = joint_population_path_info(norm_path)
        if helper_info["dataset"] and helper_info["isCanonicalPose"]:
            return str(helper_info["relPath"])
        if not helper_info["dataset"]:
            return norm_path
    patterns = [
        r"formalin/(saline|formaline|formalin|lidocane|lidocaine)/vid\d+/save_data_avg\.mat$",
        r"cap/(capsaicin-pdx\d+|capsaicin\d+|naive\d+)/save_data_avg\.mat$",
        r"(saline|formaline|formalin|lidocane|lidocaine|cpfull|wt)/vid\d+/save_data_avg(?:_[^/]+)?\.mat$",
        r"bone/[^/]+/save_data_avg(?:_[^/]+)?\.mat$",
        r"ran/sample\d+\.mat$",
    ]
    for pat in patterns:
        m = re.search(pat, norm_path)
        if m:
            return m.group(0)
    return rel_path



def find_manifest_paths(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    manifests: list[Path] = []
    root_manifest = root / "manifest.tsv"
    if root_manifest.is_file():
        manifests.append(root_manifest)
    manifests.extend(sorted(p for p in root.rglob("manifest.tsv") if p not in manifests))
    return manifests



def filter_file_list_by_dataset(file_list: list[str], dataset_filter: list[str]) -> list[str]:
    if not file_list or not dataset_filter:
        return file_list
    out: list[str] = []
    for rel_path in file_list:
        _, _, dataset_tag = parse_formalinsession_group_and_vid(rel_path)
        canonical = canonical_dataset_name(dataset_tag)
        if not canonical:
            rel_lower = rel_path.replace("\\", "/").lower()
            if rel_lower.startswith("ran/") or "/ran/" in rel_lower:
                canonical = "ran"
            elif rel_lower.startswith("bone/") or "/bone/" in rel_lower:
                canonical = "bone"
            elif rel_lower.startswith("cap/") or "/cap/" in rel_lower:
                canonical = "cap"
            elif any(tok in rel_lower for tok in ["/formalin/", "/saline/", "/formaline/", "/lidocane/", "/lidocaine/"]):
                canonical = "formalin"
        if canonical and canonical in dataset_filter:
            out.append(rel_path)
    return out



def group_sort_rank(group_name: str) -> int:
    key = str(group_name).strip().upper()
    order = {
        "RAN": 1,
        "CAPSAICIN": 2,
        "CAPSAICIN_PDX": 3,
        "SALINE": 4,
        "FORMALIN": 5,
        "LIDOCAINE": 6,
        "NAIVE": 7,
        "NANOSTING": 8,
        "VEHICLE": 9,
    }
    return order.get(key, 99)



def sort_file_list_by_group_and_vid(file_list: list[str]) -> list[str]:
    def key_fn(path: str) -> tuple[int, float, str]:
        grp, vid_num, _ = parse_formalinsession_group_and_vid(path)
        if not np.isfinite(vid_num):
            vid_num = 999.0
        return (group_sort_rank(grp), float(vid_num), path)

    return sorted(file_list, key=key_fn)



def discover_population_files(data_root: str | Path, dataset_filter: list[str] | None = None) -> list[str]:
    dataset_filter = dataset_filter or []
    file_list: list[str] = []
    for manifest_path in find_manifest_paths(data_root):
        manifest_dir = manifest_path.parent
        try:
            T = pd.read_csv(manifest_path, sep="\t")
            for col_name in ["dest_save_data_avg", "out_mat"]:
                if col_name not in T.columns:
                    continue
                for entry in T[col_name].astype(str).tolist():
                    rel_path = normalize_rel_path(entry, data_root, manifest_dir)
                    if rel_path:
                        file_list.append(rel_path)
        except Exception:
            pass
    if not file_list:
        root = Path(data_root)
        for pattern in ["save_data_AVG*.mat", "sample*.mat", "save_data_avg*.mat", "*_mouse*.mat"]:
            for path in root.rglob(pattern):
                rel_path = normalize_rel_path(path, data_root)
                if rel_path:
                    file_list.append(rel_path)
    seen: set[str] = set()
    uniq: list[str] = []
    for p in file_list:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    keep: list[str] = []
    for rel_path in uniq:
        full_path = Path(data_root) / rel_path
        if full_path.is_file():
            info = joint_population_path_info(rel_path)
            if info["dataset"]:
                if info["isCanonicalPose"]:
                    keep.append(rel_path)
            else:
                keep.append(rel_path)
    keep = filter_file_list_by_dataset(keep, dataset_filter)
    keep = sort_file_list_by_group_and_vid(keep)
    return keep



def resolve_data_path(rel_path: str | Path, default_root: str | Path) -> Path:
    rel_path = str(rel_path).replace("\\", "/")
    full = Path(default_root) / rel_path
    if full.is_file():
        return full
    parts = rel_path.split("/")
    if len(parts) > 1:
        tail = "/".join(parts[1:])
        full2 = Path(default_root) / tail
        if full2.is_file():
            return full2
    return full
