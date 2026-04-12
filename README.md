# Formalin population embedding, fixed-parameter bundle

This bundle supports one workflow:

1. run the embedding with fixed parameters
2. inspect the region map and CSV summaries
3. fill in the manual region label template
4. regenerate labeled figures from the completed manual label CSV

## Run the fixed-parameter pipeline

```bash
bash scripts/run_fixed_pipeline.sh /path/to/Population /path/to/outputs my_run
```

The run writes the usual MAT, PKL, and JSON outputs, plus analysis outputs under:

```text
<OUT_DIR>/<RUN_TAG>/analysis_outputs/
```

Key analysis files:

- `csv/per_file_region_counts.csv`
- `csv/per_file_region_counts_time_windows.csv`
- `csv/region_index_mapping.csv`
- `csv/manual_region_labels_template.csv`
- `csv/region_summary.csv`
- `figures/behavioral_map_with_indices.png`
- `figures/region_total_frames.png`

## Apply manual labels

Edit `manual_region_labels_template.csv`, fill in the `ManualLabel` column, then run:

```bash
bash scripts/apply_manual_labels.sh /path/to/outputs my_run /path/to/manual_region_labels.csv
```

This regenerates analysis artifacts from the saved run and adds:

- `csv/region_summary_manual_labels.csv`
- `figures/behavioral_map_with_manual_labels.png`
- `figures/region_total_frames_manual_labels.png`

## Direct CLI usage

```bash
python run.py \
  --data-root /path/to/Population \
  --out-dir /path/to/outputs \
  --run-tag my_run \
  --selected-mnn-k 20 \
  --selected-tsne-perplexity 32 \
  --tsne-backend pca
```

To regenerate figures from an existing run with manual labels:

```bash
python run.py \
  --out-dir /path/to/outputs \
  --run-tag my_run \
  --visualization-only \
  --manual-label-csv /path/to/manual_region_labels.csv
```
