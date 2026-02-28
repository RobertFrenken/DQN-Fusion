---
name: run-tests
description: Run the pytest test suite with optional pattern filtering
---

Run the project test suite.

## Arguments

`$ARGUMENTS` — Optional test file path or pytest `-k` pattern. If empty, runs all tests.

## Execution Steps

1. **Run pytest** with the uv venv Python:
   ```bash
   cd /users/PAS2022/rf15/KD-GAT && source .venv/bin/activate && python -m pytest tests/ -v $ARGUMENTS 2>&1
   ```

   If `$ARGUMENTS` contains a file path (ends in `.py`), pass it as the test target:
   ```bash
   cd /users/PAS2022/rf15/KD-GAT && source .venv/bin/activate && python -m pytest $ARGUMENTS -v 2>&1
   ```

   If `$ARGUMENTS` is a keyword pattern (no `.py`), use `-k`:
   ```bash
   cd /users/PAS2022/rf15/KD-GAT && source .venv/bin/activate && python -m pytest tests/ -v -k "$ARGUMENTS" 2>&1
   ```

2. **Summarize results** in a table:
   ```
   Test Results: X passed, Y failed, Z skipped

   Failed tests (if any):
   - test_name: brief reason
   ```

3. **If tests fail**, read the relevant test file and source file to diagnose the issue. Suggest a fix if the cause is clear.

## SLURM Dispatch (heavy tests)

Tests marked `@pytest.mark.slurm` (E2E pipeline, smoke training) are auto-skipped on
login nodes. To run them on a compute node:

```bash
bash scripts/run_tests_slurm.sh                         # all tests including slurm-marked
bash scripts/run_tests_slurm.sh -k "test_full_pipeline"  # specific heavy test
bash scripts/run_tests_slurm.sh -m slurm                 # only slurm-marked tests
```

## Notes

- Preprocessing tests are slow (they build actual graphs). Use `-k "not preprocessing"` to skip them.
- Layer boundary tests (`tests/test_layer_boundaries.py`) verify the 3-layer import hierarchy.
- E2E tests have a known pre-existing assertion issue with config.json.
- Heavy tests (`@pytest.mark.slurm`) auto-skip on login nodes. Submit via `scripts/run_tests_slurm.sh`.
