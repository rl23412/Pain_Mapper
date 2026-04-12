from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_pipeline(tmp_path: Path) -> None:
    data_root = tmp_path / 'Population'
    out_dir = tmp_path / 'outputs'

    gen = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'examples' / 'generate_synthetic_population.py'),
            '--output-root',
            str(data_root),
            '--frames',
            '60',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert data_root.exists(), gen.stdout + gen.stderr

    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'run_formalin_population_embed.py'),
            '--data-root',
            str(data_root),
            '--out-dir',
            str(out_dir),
            '--sample-stride',
            '6',
            '--num-per-dataset',
            '4',
            '--selected-mnn-k',
            '3',
            '--selected-tsne-perplexity',
            '5',
            '--random-seed',
            '0',
            '--tsne-backend',
            'pca',
            '--max-files',
            '4',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = out_dir / 'population_cap_formalin_joint'
    assert run_dir.exists(), run.stdout + run.stderr
    assert (run_dir / 'complete_embedding_results_population_cap_formalin_joint.pkl').exists()
    assert (run_dir / 'complete_embedding_results_population_cap_formalin_joint.mat').exists()
    assert (run_dir / 'fixed_parameters.json').exists()
    assert (run_dir / 'analysis_outputs' / 'csv' / 'per_file_region_counts.csv').exists()
    assert (run_dir / 'analysis_outputs' / 'csv' / 'manual_region_labels_template.csv').exists()
    assert (run_dir / 'analysis_outputs' / 'csv' / 'video_summary_for_python.csv').exists()
    assert (run_dir / 'analysis_outputs' / 'figures' / 'behavioral_map_with_indices.png').exists()
    assert (run_dir / 'analysis_outputs' / 'figures' / 'region_total_frames.png').exists()
