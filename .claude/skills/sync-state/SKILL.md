---
name: sync-state
description: Update STATE.md from current pipeline outputs and display it
---

Regenerate STATE.md from current experiment outputs and project DB.

## Execution Steps

1. **Run state sync**:
   ```bash
   PYTHONPATH=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT \
     /users/PAS2022/rf15/.conda/envs/gnn-experiments/bin/python -m pipeline.state_sync update 2>&1
   ```

2. **Read the updated STATE.md**:
   ```
   .claude/system/STATE.md
   ```

3. **Display the updated state** to the user with a brief summary of what changed (if anything is notable — new completed runs, failed runs, metric changes).

## Notes

- STATE.md is the primary context file for session awareness.
- Run this at the start of each session or after significant pipeline activity.
- The state_sync module reads from `experimentruns/` filesystem and `data/project.db`.
