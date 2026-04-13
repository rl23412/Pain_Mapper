# Validation notes

The cleaned bundle was revalidated after the fixed-parameter/manual-label rewrite.

Checks run on this bundle:

- `python -m compileall src examples tests`
- `PYTHONPATH=src pytest -q`
- an end-to-end smoke run of the fixed-parameter CLI

Latest local result:

- `4 passed in 21.52s`

What this validation supports:

- the package imports cleanly
- the fixed-parameter pipeline writes MAT, PKL, JSON, CSV, and PNG outputs
- the manual label template is emitted
- labeled figures can be regenerated from a completed manual label CSV
