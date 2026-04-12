from __future__ import annotations

import json
from pathlib import Path

from .config import build_arg_parser, build_pipeline_config
from .pipeline import run_pipeline



def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = build_pipeline_config(args)
    results = run_pipeline(cfg)
    summary = {
        'run_tag': cfg.run_tag,
        'run_output_dir': str(cfg.run_output_dir),
        'num_training_files': len(results.get('training_files', [])),
        'num_reembedded_files': len(results.get('reembedding_files', [])),
        'analysis_outputs_dir': str(cfg.analysis_outputs_dir) if cfg.analysis_enabled else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
