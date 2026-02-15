# Mode: Research

**Active mode**: Research — Scientific investigation, literature review, and hypothesis testing.

## Focus Areas

- OOD generalization collapse (Bug 3.6) — threshold calibration, domain shift analysis
- JumpReLU for adversarial training — sparse activation potential in GAT layers
- Cascading knowledge distillation — multi-stage KD beyond single teacher-student
- Graph neural network architecture exploration
- Benchmark comparisons with published CAN IDS methods
- Statistical analysis of experiment results

## Active Tools

- **Consensus** (`mcp__claude_ai_Consensus__search`) — Search peer-reviewed papers
- **WebFetch** — Retrieve arxiv papers, blog posts, research summaries
- **WebSearch** — Find recent publications and benchmarks

## Open Research Questions

### Priority 1: OOD Generalization (Bug 3.6)
- Why does the model collapse on out-of-distribution datasets?
- Is the issue in graph construction (temporal windows) or model capacity?
- Would domain-adaptive normalization help?
- Relevant search terms: "graph neural network domain adaptation", "CAN bus transfer learning"

### Priority 2: JumpReLU for Adversarial Training
- Can JumpReLU activation improve adversarial robustness in GAT?
- Does sparse activation reduce false positives on benign CAN traffic?
- Relevant papers: JumpReLU SAE (Anthropic), sparse autoencoders in security

### Priority 3: Cascading KD
- Can we chain KD through VGAE → GAT → DQN instead of parallel compression?
- Would intermediate representations transfer better than final logits?
- Relevant search terms: "cascaded knowledge distillation", "progressive distillation"

### Other Questions
- Custom CUDA/Triton kernels for graph attention — would they help with V100 memory?
- Feature importance analysis — which CAN signal features matter most?

## Context Files

- `src/models/` — Current architecture implementations
- `experimentruns/` — Existing results for comparison
- `USER_NOTES.md` — User's open questions and concerns

## Suppressed Topics

Do NOT initiate discussion about:
- Pipeline configuration or Snakemake debugging
- Permission rules or Claude settings
- Infrastructure setup
- Unless the user explicitly asks
