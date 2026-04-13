from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_smoke() -> None:
    run = subprocess.run(
        [sys.executable, str(ROOT / 'run_formalin_population_embed.py'), '--help'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert 'Fixed-parameter population embedding' in run.stdout
