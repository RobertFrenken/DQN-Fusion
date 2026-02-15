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
- **Project DB** — Query metrics for results tables

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
# Get metrics for results table
python -m pipeline.analytics leaderboard --metric f1 --top 20
python -m pipeline.analytics compare <run_a> <run_b>

# Get model sizes for compression ratio table
python -m pipeline.analytics memory --model vgae
python -m pipeline.analytics memory --model gat

# Dataset statistics
python -m pipeline.analytics dataset hcrl_sa
```

## Suppressed Topics

Do NOT initiate discussion about:
- Code changes, refactoring, or config modifications
- Pipeline execution or SLURM job management
- Debugging training failures
- Unless the user explicitly asks
