from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from formalin_population_embed_py.analysis import create_analysis_outputs


def test_analysis_outputs_emit_manual_label_template_and_labeled_figures(tmp_path: Path) -> None:
    results = {
        'D': np.ones((3, 3), dtype=float),
        'LL2': np.array(
            [
                [1, 1, 0],
                [1, 2, 2],
                [0, 2, 2],
            ],
            dtype=int,
        ),
        'xx': np.array([-1.0, 0.0, 1.0], dtype=float),
        'boundaryPolys': [],
        'zEmbeddings_all': [np.array([[-1.0, -1.0], [1.0, 1.0], [0.9, 0.8]], dtype=float)],
        'reembedding_labels_all': ['formalin_mouse04.mat'],
    }

    ll2_before = results['LL2'].copy()
    outputs = create_analysis_outputs(results, tmp_path / 'analysis_outputs', sampling_freq=30.0)
    assert outputs['manual_region_labels_template'].exists()
    assert outputs['region_index_mapping'].exists()
    assert outputs['behavioral_map_with_indices'].exists()

    template = pd.read_csv(outputs['manual_region_labels_template'])
    template['ManualLabel'] = template['ManualLabel'].astype(str)
    template.loc[template['CompactRegion'] == 1, 'ManualLabel'] = 'grooming'
    template.loc[template['CompactRegion'] == 2, 'ManualLabel'] = 'guarding'
    manual_csv = tmp_path / 'manual_labels.csv'
    template.to_csv(manual_csv, index=False)

    labeled_outputs = create_analysis_outputs(
        results,
        tmp_path / 'analysis_outputs_labeled',
        sampling_freq=30.0,
        manual_label_csv=manual_csv,
    )
    assert labeled_outputs['behavioral_map_with_manual_labels'].exists()
    assert labeled_outputs['region_summary_manual_labels'].exists()

    labeled_summary = pd.read_csv(labeled_outputs['region_summary_manual_labels'])
    assert 'ManualLabel' in labeled_summary.columns
    assert set(labeled_summary['ManualLabel'].fillna('').tolist()) >= {'grooming', 'guarding'}

    unlabeled_summary = pd.read_csv(outputs['region_summary'])
    labeled_base_summary = labeled_summary.drop(columns=['ManualLabel', 'Notes'])
    pd.testing.assert_frame_equal(
        unlabeled_summary.sort_index(axis=1),
        labeled_base_summary.sort_index(axis=1),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(outputs['region_index_mapping']).sort_index(axis=1),
        pd.read_csv(labeled_outputs['region_index_mapping']).sort_index(axis=1),
        check_dtype=False,
    )
    np.testing.assert_array_equal(results['LL2'], ll2_before)
