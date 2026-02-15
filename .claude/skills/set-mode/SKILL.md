---
name: set-mode
description: Switch Claude's focus mode (mlops, research, writing, data)
---

Switch the active session mode to focus Claude on a specific type of work.

## Arguments

`$ARGUMENTS` should be one of: `mlops`, `research`, `writing`, `data`

## Execution Steps

1. **Validate the mode argument**. If `$ARGUMENTS` is empty or not one of the valid modes, print:
   ```
   Usage: /set-mode <mode>
   Available modes: mlops, research, writing, data

   - mlops    — Pipeline execution, Snakemake, SLURM, config, debugging
   - research — OOD generalization, JumpReLU, literature, hypotheses
   - writing  — Paper drafting, documentation, results interpretation
   - data     — Dataset ingestion, preprocessing, validation, cache
   ```
   And stop.

2. **Read the mode context file**:
   ```
   .claude/system/modes/$ARGUMENTS.md
   ```

3. **Display the mode context** to the user. Print the full contents of the mode file so it becomes part of the conversation context.

4. **Confirm activation** with a brief summary:
   ```
   Mode switched to: <mode name>
   Focus: <1-line summary from the mode file>
   Suppressed: <suppressed topics>
   ```

## Notes

- The mode switch works by injecting the mode context into the conversation. There is no persistent state — each new session starts fresh.
- You can switch modes mid-session by running `/set-mode` again.
- The mode file content instructs Claude on what to focus on and what to suppress.
