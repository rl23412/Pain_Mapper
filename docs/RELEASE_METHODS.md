# Fixed-parameter workflow

This release bundle uses one path.

The embedding is built from pose-derived features with fixed feature weights, fixed MNN selection, and fixed t-SNE settings supplied on the command line or left at the bundle defaults.

The post-run workflow is:

1. build the embedding and watershed regions
2. export region index summaries and figures
3. export a blank manual label template for compact region IDs
4. re-run analysis in visualization-only mode with a filled manual label CSV to generate labeled figures
