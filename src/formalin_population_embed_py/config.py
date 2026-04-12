from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    sval = str(value).strip().lower()
    if sval in {"1", "true", "yes", "y", "on"}:
        return True
    if sval in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_dataset_filter(filter_raw: Any) -> list[str]:
    if not filter_raw:
        return []
    if isinstance(filter_raw, (list, tuple)):
        tokens = [str(x) for x in filter_raw]
    else:
        tokens = str(filter_raw).replace(";", ",").replace(" ", ",").split(",")
    out: list[str] = []
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        canonical = canonical_dataset_name(token)
        if canonical == "all":
            return ["cap", "formalin"]
        if canonical and canonical not in out:
            out.append(canonical)
    return out


def canonical_dataset_name(token: str | None) -> str:
    if not token:
        return ""
    t = str(token).strip().lower()
    if t in {"all", "*"}:
        return "all"
    if t in {"cap", "capsaicin", "capsaicin-pdx", "capsaicin_pdx", "cap_naive"}:
        return "cap"
    if t in {"formalin", "formaline", "saline", "lidocane", "lidocaine", "wt", "cpfull"}:
        return "formalin"
    if t in {"ran"}:
        return "ran"
    if t in {"bone", "naive", "nanosting", "nano-sting", "vehicle"}:
        return "bone"
    return ""


@dataclass
class FixedPairwisePriorConfig:
    enable: bool = False
    snout_ear_gain: float = 0.25
    ear_ear_gain: float = 0.35


@dataclass
class ScientificConfig:
    enabled: bool = False
    control_anchor_minutes: float = 20.0
    acute_pain_minutes: float = 5.0
    shared_non_pain_groups: tuple[str, ...] = (
        "SALINE",
        "CAP_NAIVE",
        "NAIVE",
        "LIDOCAINE",
        "CAPSAICIN_PDX",
    )
    control_anchor_groups: tuple[str, ...] = ("SALINE", "CAP_NAIVE", "NAIVE")
    acute_pain_groups: tuple[str, ...] = ("FORMALIN", "CAPSAICIN")
    enable_mnn_state_aware: bool = False
    prefer_anchor_reference: bool = True
    include_acute_pain_anchor: bool = True
    min_state_points: int = 25
    supervision_enable: bool = False
    supervision_mode: str = "both"
    enable_pain_diagnostics: bool = False
    write_orientation_qc: bool = True
    write_segment_qc: bool = True
    write_rigid_qc: bool = True
    pca_modes_override: int | None = None
    fixed_pairwise_prior: FixedPairwisePriorConfig = field(default_factory=FixedPairwisePriorConfig)


@dataclass
class SupervisionConfig:
    enable: bool = False
    mode: str = "both"
    groups: tuple[str, ...] = ("NONPAINFUL", "PAINFUL")
    phase_windows: tuple[tuple[float, float], ...] = (
        (0.0, 5.0),
        (5.0, 15.0),
        (15.0, 30.0),
        (30.0, float("inf")),
    )
    phase_names: tuple[str, ...] = (
        "PainWindow_0_5",
        "Post_5_15",
        "Post_15_30",
        "Post_30_plus",
    )
    n_lda: int = 8
    class_balance: str = "balanced"
    feature_aug_group_weight: float = 0.2
    feature_aug_phase_weight: float = 0.2
    run_region12_classifier: bool = False
    run_semi_supervised: bool = False
    run_pain_diagnostics: bool = False


@dataclass
class RunParameters:
    sampling_freq: float = 30.0
    min_f: float = 0.2
    max_f: float = 12.0
    num_periods: int = 10
    omega0: float = 5.0
    pca_modes: int = 20
    kd_neighbors: int = 5
    min_template_length: int = 1
    perplexity: float = 32.0
    rel_tol: float = 1e-4
    sigma_tolerance: float = 1e-5
    max_neighbors: int = 200
    batch_size: int = 10000
    max_optim_iter: int = 25
    projection_refine: bool = False
    training_rel_tol: float = 2e-3
    training_perplexity: float = 20.0
    training_num_points: int = 10000


@dataclass
class PipelineConfig:
    data_root: Path
    out_dir: Path
    run_tag: str = "population_cap_formalin_joint"
    datasets_include: list[str] = field(default_factory=lambda: ["cap", "formalin"])
    pca_pre_frame_cap: int | float = 30000
    parallel_enable: bool = True
    parallel_workers: int = 1
    parameters: RunParameters = field(default_factory=RunParameters)
    scientific: ScientificConfig = field(default_factory=ScientificConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    use_mnn_correction: bool = False
    selected_mnn_k: int = 20
    selected_tsne_perplexity: float | None = None
    sample_stride: int = 20
    num_per_dataset: int = 320
    visualization_only: bool = False
    random_seed: int = 0
    save_mat_files: bool = True
    save_pickle_files: bool = True
    analysis_enabled: bool = True
    tsne_backend: str = "pca"
    max_files: int | None = None
    manual_label_csv: Path | None = None

    @property
    def run_output_dir(self) -> Path:
        return self.out_dir / self.run_tag

    @property
    def analysis_outputs_dir(self) -> Path:
        return self.run_output_dir / "analysis_outputs"

    @property
    def analysis_outputs_semi_dir(self) -> Path:
        return self.run_output_dir / "analysis_outputs_semi"


def scientific_preset_from_env() -> ScientificConfig:
    enabled = parse_bool(os.getenv("SCIENTIFIC_PAIN_PRESET"), False)
    cfg = ScientificConfig(
        enabled=enabled,
        control_anchor_minutes=float(os.getenv("CONTROL_ANCHOR_MINUTES") or 20),
        acute_pain_minutes=float(os.getenv("ACUTE_PAIN_MINUTES") or 5),
        enable_mnn_state_aware=parse_bool(os.getenv("MNN_STATE_AWARE"), enabled),
        prefer_anchor_reference=parse_bool(os.getenv("MNN_PREFER_ANCHOR_REFERENCE"), True),
        include_acute_pain_anchor=parse_bool(os.getenv("MNN_INCLUDE_ACUTE_PAIN_ANCHOR"), True),
        min_state_points=max(5, int(float(os.getenv("MNN_MIN_STATE_POINTS") or 25))),
        supervision_enable=parse_bool(os.getenv("SUPERVISION_ENABLE"), enabled),
        supervision_mode=(os.getenv("SUPERVISION_MODE") or "both").strip() or "both",
        enable_pain_diagnostics=parse_bool(os.getenv("PAIN_DIAGNOSTICS_ENABLE"), enabled),
        write_orientation_qc=parse_bool(os.getenv("ORIENTATION_QC_ENABLE"), True),
        write_segment_qc=parse_bool(os.getenv("SEGMENT_QC_ENABLE"), True),
        write_rigid_qc=parse_bool(os.getenv("RIGID_SEGMENT_QC_ENABLE"), True),
        pca_modes_override=(
            int(float(os.getenv("N_PCA"))) if os.getenv("N_PCA") not in {None, "", "nan", "NaN"} else None
        ),
        fixed_pairwise_prior=FixedPairwisePriorConfig(
            enable=parse_bool(os.getenv("PAIRWISE_CRANIAL_PRIOR_ENABLE"), enabled),
            snout_ear_gain=float(os.getenv("NOSE_EYE_PAIR_GAIN") or 0.25),
            ear_ear_gain=float(os.getenv("EYE_EYE_PAIR_GAIN") or 0.35),
        ),
    )
    return cfg


def default_supervision(scientific_cfg: ScientificConfig) -> SupervisionConfig:
    supervision = SupervisionConfig()
    if scientific_cfg.enabled:
        supervision.enable = scientific_cfg.supervision_enable
        supervision.mode = scientific_cfg.supervision_mode
        supervision.run_pain_diagnostics = scientific_cfg.enable_pain_diagnostics
    return supervision


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_formalin_population_embed.py",
        description="Fixed-parameter population embedding with manual labeling outputs.",
    )
    p.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT") or "/hpc/group/1920-PainDetect/Population"))
    p.add_argument("--out-dir", type=Path, default=Path(os.getenv("OUT_DIR") or "./outputs"))
    p.add_argument("--run-tag", default=os.getenv("RUN_TAG") or "population_cap_formalin_joint")
    p.add_argument("--datasets-include", default=os.getenv("DATASETS_INCLUDE") or "cap,formalin")
    p.add_argument("--preset", choices=["baseline", "scientific"], default=None)
    p.add_argument("--scientific-pain-preset", action="store_true")
    p.add_argument("--pca-pre-frame-cap", default=os.getenv("PCA_PRE_FRAME_CAP") or "30000")
    p.add_argument("--parallel-enable", default=os.getenv("PARALLEL_ENABLE") or "1")
    p.add_argument("--parallel-workers", type=int, default=int(os.getenv("PARALLEL_WORKERS") or os.getenv("SLURM_CPUS_PER_TASK") or "1"))
    p.add_argument("--mnn-enable", default=os.getenv("MNN_ENABLE"), help="Override MNN_ENABLE.")
    p.add_argument("--save-mat-files", action="store_true", default=True)
    p.add_argument("--no-save-mat-files", dest="save_mat_files", action="store_false")
    p.add_argument("--save-pickle-files", action="store_true", default=True)
    p.add_argument("--no-save-pickle-files", dest="save_pickle_files", action="store_false")
    p.add_argument("--analysis-enabled", action="store_true", default=True)
    p.add_argument("--no-analysis", dest="analysis_enabled", action="store_false")
    p.add_argument("--visualization-only", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--sample-stride", type=int, default=20)
    p.add_argument("--num-per-dataset", type=int, default=320)
    p.add_argument("--selected-mnn-k", type=int, default=20)
    p.add_argument("--selected-tsne-perplexity", type=float, default=float("nan"))
    p.add_argument("--tsne-backend", choices=["pca", "sklearn"], default=None)
    p.add_argument("--manual-label-csv", type=Path, default=Path(os.getenv("MANUAL_LABEL_CSV")) if os.getenv("MANUAL_LABEL_CSV") else None)
    p.add_argument("--max-files", type=int, default=None)
    return p


def build_pipeline_config(args: argparse.Namespace | None = None) -> PipelineConfig:
    if args is None:
        args = build_arg_parser().parse_args()

    scientific = scientific_preset_from_env()
    if args.preset == "scientific" or args.scientific_pain_preset:
        scientific.enabled = True
        scientific.enable_mnn_state_aware = True if os.getenv("MNN_STATE_AWARE") is None else scientific.enable_mnn_state_aware
        scientific.supervision_enable = True if os.getenv("SUPERVISION_ENABLE") is None else scientific.supervision_enable
        scientific.enable_pain_diagnostics = True if os.getenv("PAIN_DIAGNOSTICS_ENABLE") is None else scientific.enable_pain_diagnostics
    supervision = default_supervision(scientific)

    params = RunParameters()
    if scientific.pca_modes_override:
        params.pca_modes = int(scientific.pca_modes_override)

    pca_pre_frame_cap_raw = str(args.pca_pre_frame_cap).strip().lower()
    if pca_pre_frame_cap_raw in {"all", "none", "off", "inf", "infinity"}:
        pca_pre_frame_cap: int | float = float("inf")
    else:
        pca_pre_frame_cap = int(float(pca_pre_frame_cap_raw))

    selected_tsne_perplexity = args.selected_tsne_perplexity
    if not (selected_tsne_perplexity == selected_tsne_perplexity and selected_tsne_perplexity > 0):
        selected_tsne_perplexity = None

    mnn_enable_override = args.mnn_enable
    if mnn_enable_override is None:
        use_mnn_correction = scientific.enabled or scientific.enable_mnn_state_aware
    else:
        use_mnn_correction = parse_bool(mnn_enable_override, scientific.enabled or scientific.enable_mnn_state_aware)

    tsne_backend = args.tsne_backend or os.getenv("TSNE_BACKEND")
    if not tsne_backend:
        tsne_backend = "sklearn" if scientific.enabled else "pca"

    cfg = PipelineConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        run_tag=args.run_tag,
        datasets_include=parse_dataset_filter(args.datasets_include),
        pca_pre_frame_cap=pca_pre_frame_cap,
        parallel_enable=parse_bool(args.parallel_enable, True),
        parallel_workers=max(1, int(args.parallel_workers)),
        parameters=params,
        scientific=scientific,
        supervision=supervision,
        use_mnn_correction=use_mnn_correction,
        selected_mnn_k=args.selected_mnn_k,
        selected_tsne_perplexity=selected_tsne_perplexity,
        sample_stride=max(1, int(args.sample_stride)),
        num_per_dataset=max(1, int(args.num_per_dataset)),
        visualization_only=bool(args.visualization_only),
        random_seed=int(args.random_seed),
        save_mat_files=bool(args.save_mat_files),
        save_pickle_files=bool(args.save_pickle_files),
        analysis_enabled=bool(args.analysis_enabled),
        tsne_backend=str(tsne_backend),
        max_files=args.max_files,
        manual_label_csv=args.manual_label_csv,
    )
    return cfg
