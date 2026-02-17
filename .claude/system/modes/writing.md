# Mode: Writing

**Active mode**: Writing — Paper drafting, documentation, and results presentation.

## Focus Areas

- Research paper drafting (results, methodology, related work sections)
- Lab documentation via MkDocs site (osu-car-msl.github.io/lab-setup-guide/)
- Results interpretation and visualization descriptions
- Figure captions and table formatting
- Literature citations and bibliography management

## Active Resources

- **GitHub** (`gh` CLI) — Query papers repo, lab-setup-guide repo
- **WebFetch** — Access osu-car-msl.github.io for existing documentation
- **Consensus** — Find citations for claims and comparisons
- **W&B** — Query metrics for results tables

## Writing Tasks

### Paper Sections
- Abstract and introduction framing
- Methodology: 3-stage pipeline description (VGAE → GAT → DQN)
- Results: performance tables, KD compression ratios, per-dataset analysis
- Related work: CAN bus IDS, knowledge distillation, GNN for security

### Documentation
- MkDocs site updates for lab-setup-guide
- Experiment documentation and reproduction guides
- README updates for the project

## Useful Queries

```bash
# Get metrics from experiment runs
python -m pipeline.export   # Generates leaderboard, metrics JSON

# Browse W&B project for detailed metrics
# https://wandb.ai/ → project kd-gat

# Export dashboard data for visualization
bash scripts/export_dashboard.sh --dry-run
```

## Suppressed Topics

Do NOT initiate discussion about:
- Code changes, refactoring, or config modifications
- Pipeline execution or SLURM job management
- Debugging training failures
- Unless the user explicitly asks
