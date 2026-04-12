from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import parse_formalinsession_group_and_vid


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def _grid_index_from_points(points: np.ndarray, xx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    xx = np.asarray(xx, dtype=float).reshape(-1)
    if points.size == 0 or xx.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    grid_size = int(xx.size)
    map_min = float(np.min(xx))
    map_max = float(np.max(xx))
    if not np.isfinite(map_min) or not np.isfinite(map_max) or map_max <= map_min:
        map_min = -1.0
        map_max = 1.0
    scale = float(grid_size - 1) / float(map_max - map_min)
    gx = np.round((points[:, 0] - map_min) * scale + 1.0).astype(int)
    gy = np.round((points[:, 1] - map_min) * scale + 1.0).astype(int)
    gx[gx < 1] = 1
    gy[gy < 1] = 1
    gx[gx > grid_size] = grid_size
    gy[gy > grid_size] = grid_size
    return gx - 1, gy - 1


def _map_compact_regions(reg_ids_old: np.ndarray, valid_region_ids: list[int]) -> np.ndarray:
    region_map = {int(old): idx + 1 for idx, old in enumerate(valid_region_ids)}
    reg_ids_old = np.asarray(reg_ids_old, dtype=int).reshape(-1)
    out = np.zeros_like(reg_ids_old, dtype=int)
    for old, new in region_map.items():
        out[reg_ids_old == old] = new
    return out


def _count_regions(reg_ids: np.ndarray, num_regions: int) -> np.ndarray:
    reg_ids = np.asarray(reg_ids, dtype=int).reshape(-1)
    valid = reg_ids[(reg_ids >= 1) & (reg_ids <= num_regions)]
    counts = np.bincount(valid, minlength=num_regions + 1)
    return counts[1 : num_regions + 1]


def _region_centroids_from_ll2(ll2: np.ndarray, valid_region_ids: list[int]) -> dict[int, tuple[float, float]]:
    ll2 = np.asarray(ll2)
    out: dict[int, tuple[float, float]] = {}
    for old_id in valid_region_ids:
        pix = np.argwhere(ll2 == old_id)
        if pix.size == 0:
            out[int(old_id)] = (float('nan'), float('nan'))
        else:
            yx = np.mean(pix, axis=0)
            out[int(old_id)] = (float(yx[1]), float(yx[0]))
    return out


def _normalize_manual_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(' ', '_')
        if key in {'compactregion', 'compact_region', 'region', 'region_id'}:
            rename_map[col] = 'CompactRegion'
        elif key in {'originalwatershedregion', 'original_watershed_region'}:
            rename_map[col] = 'OriginalWatershedRegion'
        elif key in {'manuallabel', 'manual_label', 'label'}:
            rename_map[col] = 'ManualLabel'
        elif key in {'notes', 'note'}:
            rename_map[col] = 'Notes'
    out = df.rename(columns=rename_map).copy()
    if 'CompactRegion' not in out.columns:
        raise ValueError('Manual label CSV must contain a CompactRegion column.')
    if 'ManualLabel' not in out.columns:
        out['ManualLabel'] = ''
    if 'Notes' not in out.columns:
        out['Notes'] = ''
    out['CompactRegion'] = pd.to_numeric(out['CompactRegion'], errors='coerce').astype('Int64')
    out = out.dropna(subset=['CompactRegion']).copy()
    out['CompactRegion'] = out['CompactRegion'].astype(int)
    return out


def _build_manual_label_template(map_table: pd.DataFrame) -> pd.DataFrame:
    template = map_table.copy()
    template['ManualLabel'] = ''
    template['Notes'] = ''
    return template[['CompactRegion', 'OriginalWatershedRegion', 'ManualLabel', 'Notes']]


def _manual_label_lookup(manual_df: pd.DataFrame | None) -> dict[int, str]:
    if manual_df is None or manual_df.empty:
        return {}
    out: dict[int, str] = {}
    for row in manual_df.itertuples(index=False):
        label = str(getattr(row, 'ManualLabel', '') or '').strip()
        if label:
            out[int(row.CompactRegion)] = label
    return out


def _save_behavioral_map(
    results: dict[str, Any],
    valid_region_ids: list[int],
    fig_path: Path,
    region_text: dict[int, str],
    title: str,
) -> None:
    D = np.asarray(results['D'], dtype=float)
    ll2 = np.asarray(results['LL2'], dtype=int)
    boundary_polys = _as_list(results.get('boundaryPolys', []))
    centroids = _region_centroids_from_ll2(ll2, valid_region_ids)
    region_map = {int(old): idx + 1 for idx, old in enumerate(valid_region_ids)}

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(D, cmap='gray_r', origin='upper')
    for poly in boundary_polys:
        poly = np.asarray(poly, dtype=float)
        if poly.ndim == 2 and poly.shape[1] == 2 and poly.shape[0] > 1:
            ax.plot(poly[:, 1], poly[:, 0], linewidth=1.0)
    for old_id, (cx, cy) in centroids.items():
        if np.isfinite(cx) and np.isfinite(cy):
            compact_id = region_map[old_id]
            text = region_text.get(compact_id, str(compact_id))
            ax.text(
                cx,
                cy,
                text,
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.72, linewidth=0),
            )
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def _save_region_totals_plot(region_summary: pd.DataFrame, fig_path: Path, label_lookup: dict[int, str] | None = None) -> None:
    label_lookup = label_lookup or {}
    plot_df = region_summary.copy()
    x_labels = [label_lookup.get(int(r), str(int(r))) for r in plot_df['CompactRegion'].tolist()]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    ax.bar(np.arange(plot_df.shape[0]), plot_df['TotalFrames'].to_numpy(dtype=float))
    ax.set_xticks(np.arange(plot_df.shape[0]))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel('Region')
    ax.set_ylabel('Total frames')
    ax.set_title('Region occupancy totals')
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)


def create_analysis_outputs(
    results: dict[str, Any],
    output_root: str | Path,
    sampling_freq: float = 30.0,
    manual_label_csv: str | Path | None = None,
) -> dict[str, Path]:
    output_root = Path(output_root)
    csv_dir = output_root / 'csv'
    fig_dir = output_root / 'figures'
    frame_indices_dir = csv_dir / 'frame_indices_per_video'
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    frame_indices_dir.mkdir(parents=True, exist_ok=True)

    z_all = _as_list(results.get('zEmbeddings_all', []))
    file_names = [str(x) for x in _as_list(results.get('reembedding_labels_all', []))]
    ll2 = np.asarray(results['LL2'], dtype=int)
    xx = np.asarray(results['xx'], dtype=float).reshape(-1)

    valid_region_ids = sorted(int(v) for v in np.unique(ll2[ll2 > 0]).tolist())
    num_regions = len(valid_region_ids)
    region_map = {int(old): idx + 1 for idx, old in enumerate(valid_region_ids)}

    time_window_defs = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 35.0], [35.0, np.inf]], dtype=float)
    time_window_labels = ['0-10 min', '10-20 min', '20-35 min', '35+ min']

    per_file_rows: list[dict[str, Any]] = []
    per_file_time_rows: list[dict[str, Any]] = []
    video_summary_rows: list[dict[str, Any]] = []
    region_totals = np.zeros((num_regions,), dtype=float)
    region_video_counts = np.zeros((num_regions,), dtype=float)

    for i, file_name in enumerate(file_names):
        z = np.asarray(z_all[i], dtype=float) if i < len(z_all) else np.zeros((0, 2), dtype=float)
        gx, gy = _grid_index_from_points(z, xx)
        wr_old = np.zeros((z.shape[0],), dtype=int)
        valid = (gx >= 0) & (gx < ll2.shape[1]) & (gy >= 0) & (gy < ll2.shape[0])
        wr_old[valid] = ll2[gy[valid], gx[valid]].astype(int)
        wr = _map_compact_regions(wr_old, valid_region_ids)
        counts = _count_regions(wr, num_regions)
        region_totals += counts
        region_video_counts += counts > 0

        group, vid_num, dataset_tag = parse_formalinsession_group_and_vid(file_name)
        row: dict[str, Any] = {
            'VideoIndex': i + 1,
            'File': file_name,
            'Group': group or 'UNKNOWN',
            'Dataset': dataset_tag or 'unknown',
            'VideoId': vid_num,
            'Frames': int(z.shape[0]),
            'ValidFrames': int(np.sum(wr > 0)),
        }
        for region_idx in range(num_regions):
            row[f'Region_{region_idx + 1}'] = int(counts[region_idx])
        per_file_rows.append(row)

        frame_times_min = np.arange(z.shape[0], dtype=float) / (float(sampling_freq) * 60.0)
        for tw_idx, (tw_start, tw_end) in enumerate(time_window_defs):
            if np.isinf(tw_end):
                mask = frame_times_min >= tw_start
            else:
                mask = (frame_times_min >= tw_start) & (frame_times_min < tw_end)
            counts_tw = _count_regions(wr[mask], num_regions) if mask.size else np.zeros((num_regions,), dtype=int)
            tw_row: dict[str, Any] = {
                'VideoIndex': i + 1,
                'File': file_name,
                'Group': group or 'UNKNOWN',
                'Dataset': dataset_tag or 'unknown',
                'TimeWindow': time_window_labels[tw_idx],
                'Frames': int(np.sum(mask)),
            }
            for region_idx in range(num_regions):
                tw_row[f'Region_{region_idx + 1}'] = int(counts_tw[region_idx])
            per_file_time_rows.append(tw_row)

        max_frames = int(np.max(counts)) if counts.size else 0
        max_frames = max(max_frames, 1)
        region_frame_table = pd.DataFrame()
        for region_idx in range(1, num_regions + 1):
            frame_idx = np.flatnonzero(wr == region_idx) + 1
            padded = np.full((max_frames,), np.nan, dtype=float)
            if frame_idx.size:
                padded[: frame_idx.size] = frame_idx.astype(float)
            region_frame_table[f'Region_{region_idx}'] = padded
        sanitized_file_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in file_name)
        frame_csv = frame_indices_dir / f'{i + 1:03d}_{sanitized_file_name}_frame_indices.csv'
        region_frame_table.to_csv(frame_csv, index=False)

        gx, gy = _grid_index_from_points(z, xx)
        valid_xy = (gx >= 0) & (gx < ll2.shape[1]) & (gy >= 0) & (gy < ll2.shape[0])
        video_summary_rows.append(
            {
                'VideoIndex': i + 1,
                'VideoFileName': file_name,
                'Group': group or 'UNKNOWN',
                'Dataset': dataset_tag or 'unknown',
                'TotalFrames': int(z.shape[0]),
                'ValidFrames': int(np.sum(valid_xy)),
                'RegionsWithFrames': int(np.sum(counts > 0)),
            }
        )

    map_rows = []
    for old_id, compact_id in region_map.items():
        map_rows.append({'CompactRegion': compact_id, 'OriginalWatershedRegion': int(old_id)})
    map_table = pd.DataFrame(map_rows)

    region_summary_rows = []
    total_behavioral_frames = float(np.sum(region_totals))
    for old_id in valid_region_ids:
        compact_id = region_map[old_id]
        total_frames = int(region_totals[compact_id - 1])
        region_summary_rows.append(
            {
                'CompactRegion': compact_id,
                'OriginalWatershedRegion': int(old_id),
                'TotalFrames': total_frames,
                'FractionOfBehavioralFrames': (total_frames / total_behavioral_frames) if total_behavioral_frames > 0 else 0.0,
                'VideosWithFrames': int(region_video_counts[compact_id - 1]),
            }
        )
    region_summary = pd.DataFrame(region_summary_rows)

    per_file_df = pd.DataFrame(per_file_rows)
    per_file_time_df = pd.DataFrame(per_file_time_rows)
    video_summary_df = pd.DataFrame(video_summary_rows)
    transposed_df = per_file_df.set_index('File').T if not per_file_df.empty else pd.DataFrame()

    per_file_csv = csv_dir / 'per_file_region_counts.csv'
    per_file_df.to_csv(per_file_csv, index=False)
    per_file_time_csv = csv_dir / 'per_file_region_counts_time_windows.csv'
    per_file_time_df.to_csv(per_file_time_csv, index=False)
    region_index_csv = csv_dir / 'region_index_mapping.csv'
    map_table.to_csv(region_index_csv, index=False)
    region_summary_csv = csv_dir / 'region_summary.csv'
    region_summary.to_csv(region_summary_csv, index=False)
    transposed_csv = csv_dir / 'region_counts_transposed.csv'
    transposed_df.to_csv(transposed_csv)
    video_summary_csv = csv_dir / 'video_summary_for_python.csv'
    video_summary_df.to_csv(video_summary_csv, index=False)

    manual_template = _build_manual_label_template(map_table)
    manual_template_csv = csv_dir / 'manual_region_labels_template.csv'
    manual_template.to_csv(manual_template_csv, index=False)

    if num_regions > 0:
        index_text = {int(r): str(int(r)) for r in map_table['CompactRegion'].tolist()}
        _save_behavioral_map(results, valid_region_ids, fig_dir / 'behavioral_map_with_indices.png', index_text, 'Behavioral map with region indices')
        _save_region_totals_plot(region_summary, fig_dir / 'region_total_frames.png')

    manual_df: pd.DataFrame | None = None
    labeled_summary_csv: Path | None = None
    manual_map_fig: Path | None = None
    manual_totals_fig: Path | None = None
    if manual_label_csv is not None:
        manual_path = Path(manual_label_csv)
        if not manual_path.exists():
            raise FileNotFoundError(f'Manual label CSV not found: {manual_path}')
        manual_df = _normalize_manual_label_columns(pd.read_csv(manual_path))
        manual_df = map_table.merge(manual_df[['CompactRegion', 'ManualLabel', 'Notes']], on='CompactRegion', how='left')
        manual_df['ManualLabel'] = manual_df['ManualLabel'].fillna('')
        manual_df['Notes'] = manual_df['Notes'].fillna('')
        label_lookup = _manual_label_lookup(manual_df)
        labeled_summary = region_summary.merge(manual_df[['CompactRegion', 'ManualLabel', 'Notes']], on='CompactRegion', how='left')
        labeled_summary_csv = csv_dir / 'region_summary_manual_labels.csv'
        labeled_summary.to_csv(labeled_summary_csv, index=False)
        if num_regions > 0:
            map_text = {int(r): f"{int(r)}\n{label_lookup[r]}" if r in label_lookup else str(int(r)) for r in map_table['CompactRegion'].tolist()}
            manual_map_fig = fig_dir / 'behavioral_map_with_manual_labels.png'
            _save_behavioral_map(results, valid_region_ids, manual_map_fig, map_text, 'Behavioral map with manual labels')
            manual_totals_fig = fig_dir / 'region_total_frames_manual_labels.png'
            _save_region_totals_plot(region_summary, manual_totals_fig, label_lookup=label_lookup)

    outputs: dict[str, Path] = {
        'csv_dir': csv_dir,
        'fig_dir': fig_dir,
        'frame_indices_dir': frame_indices_dir,
        'per_file_region_counts': per_file_csv,
        'per_file_region_counts_time_windows': per_file_time_csv,
        'region_index_mapping': region_index_csv,
        'manual_region_labels_template': manual_template_csv,
        'region_summary': region_summary_csv,
        'region_counts_transposed': transposed_csv,
        'video_summary_for_python': video_summary_csv,
    }
    if num_regions > 0:
        outputs['behavioral_map_with_indices'] = fig_dir / 'behavioral_map_with_indices.png'
        outputs['region_total_frames'] = fig_dir / 'region_total_frames.png'
    if labeled_summary_csv is not None:
        outputs['region_summary_manual_labels'] = labeled_summary_csv
    if manual_map_fig is not None:
        outputs['behavioral_map_with_manual_labels'] = manual_map_fig
    if manual_totals_fig is not None:
        outputs['region_total_frames_manual_labels'] = manual_totals_fig
    return outputs
