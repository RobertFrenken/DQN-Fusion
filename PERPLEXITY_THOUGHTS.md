<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Currently my ML project pipeline takes in explicit arguments from the CLI, which creates a slurm script for each job in the pipeline, and then the training python files takes in the arguments provided by the slurm script. Based off all the configurations and hierarchy of the configurations, the model that is trained and the logs go into that specific folder. It has taken alot of time, is fragile, but is the best solution I can think of right now. Are there better ways to do this have been done in other ML OPs or AI researcher projects?

The pattern you describe—CLI arguments → script generator → one SLURM script per job → Python entrypoint parsing those args → hand‑crafted directory hierarchy—is very common in “first serious version” research codebases. It works, but it is brittle because:

- The scheduler layer (SLURM scripts) is too entangled with experiment configuration.
- There is no single, authoritative record of “what was this run?” beyond filenames and ad‑hoc conventions.
- Every new variant multiplies the places you must update (CLI, script template, path logic, logging).

Most mature ML/HPC setups converge on a few recurring design patterns that simplify this substantially.

Below is a synthesis of what is widely used in research labs and MLOps projects on SLURM, plus a concrete migration path tailored to your setup.

***

## 1. Core Idea: Config File as the Only Source of Truth

The strongest simplification is to treat *a single immutable config file* as the canonical description of an experiment. Everything else—SLURM submission, Python training code, log locations—is derived from that.

In this pattern:

1. You generate a JSON/YAML config for each run (or each logical “experiment”).
2. Your SLURM script takes *only* `--config path/to/config.yaml` (and maybe a run ID) as arguments.
3. The training script loads the config and never accepts individual hyperparameters or paths from the CLI.
4. The training script *writes back* a small manifest of artifacts and metrics in the same folder.

This “frozen config” approach is exactly what tools like Hydra’s read‑only/structured configs support: you compose a configuration once and then forbid runtime modification, making runs reproducible and debuggable.[^1][^2]

### How this would look in your project

Given your tree (simplified):

```text
experimentruns/
  automotive/
    hcrl_ch/
      supervised/...           # logs, models, etc.
    hcrl_sa/
    set_01/...
    slurm_runs/
      hcrl_ch/
      hcrl_sa/
      set_01/
      ...
```

You introduce a `configs/` directory and an explicit `runs/` directory:

```text
configs/
  automotive/
    hcrl_ch/
      supervised_curriculum.yaml
      unsupervised_autoencoder.yaml
    set_01/
      ...
runs/
  automotive/
    hcrl_ch/
      2026-01-26_220101_supervised_curriculum/
        config.yaml
        manifest.json
        logs/
        checkpoints/
```

A small Python “submit” script then:

- Takes a logical experiment name (e.g. `automotive/hcrl_ch/supervised_curriculum`).
- Materializes a concrete run directory with a copied/expanded `config.yaml` and timestamped name.
- Submits a *generic* SLURM script with `--config path/to/runs/.../config.yaml`.

Your SLURM template becomes almost static:

```bash
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.log

python train.py --config "$CONFIG_PATH"
```

All the chaos that used to live in dozens of slightly different scripts now lives in structured configs and a tiny Python launcher.

***

## 2. Use Job Arrays + a Manifest Rather Than One Script per Job

One pain point you alluded to is generating a separate SLURM script per job in a pipeline. On HPC, the idiomatic alternative is:

- Maintain a *manifest* (CSV/JSON/YAML) listing all experiments.
- Use a *single* `sbatch` submission with a job array (`--array 0-N`) where each array index picks one line from the manifest.[^3][^4][^5]

This pattern is well documented in SLURM job‑array tutorials for ML workloads: they show a manifest with seeds, models, preprocessing choices, etc., and a single sbatch script that reads `SLURM_ARRAY_TASK_ID` to know which row to run.[^6][^4][^3]

### Manifest pattern sketch

`experiments.csv`:

```csv
id,experiment,config_path
0,automotive/hcrl_ch/supervised_curriculum,configs/automotive/hcrl_ch/supervised_curriculum.yaml
1,automotive/hcrl_sa/unsupervised,configs/automotive/hcrl_sa/unsupervised.yaml
2,automotive/set_01/autoencoder,configs/automotive/set_01/autoencoder.yaml
...
```

Single SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=auto_experiments
#SBATCH --array=0-2          # one per row in experiments.csv
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.log

ROW=$(($SLURM_ARRAY_TASK_ID + 1))   # skip header
LINE=$(sed -n "${ROW}p" experiments.csv)
ID=$(echo "$LINE" | cut -d, -f1)
EXPERIMENT=$(echo "$LINE" | cut -d, -f2)
CONFIG_PATH=$(echo "$LINE" | cut -d, -f3)

python submit_and_train.py \
  --experiment "$EXPERIMENT" \
  --config "$CONFIG_PATH" \
  --run_id "$ID"
```

`submit_and_train.py` then:

- Creates the run folder under `runs/` based on `EXPERIMENT` and `run_id`.
- Copies/expands `config.yaml` into that folder.
- Calls `train.py --config runs/.../config.yaml`.

This removes the need to generate a separate slurm script per experiment; the only generated artifact is the config (and maybe the manifest). It also follows cluster best practices around not “spamming the scheduler” with thousands of tiny jobs.[^4]

***

## 3. Bring in a Config System (Hydra / Pydantic / dataclasses)

Right now your “configuration system” is: CLI args + folder naming conventions. That works, but:

- It’s hard to see all options in one place.
- Refactoring is risky (renaming an argument breaks scripts and directories).
- Validation logic is scattered.

Most modern research‑grade projects adopt a config framework such as Hydra, OmegaConf, or a structured Pydantic layer. The idea:[^2][^7][^1]

- Define a small number of *structured config* dataclasses/Pydantic models (model, data, training, SLURM resources).
- Compose them from YAML files (e.g. `configs/model/resnet.yaml`, `configs/data/hcrl_ch.yaml`).
- Use overrides to derive variants (`python train.py model=resnet data=hcrl_ch trainer.max_epochs=50`).
- Once composed, freeze the config to prevent accidental mutation at runtime.[^1]

Given your earlier “ML Workflow Design Patterns” note (ConfigStore, Pydantic v2, PathResolver), you are already close to this. The key shift is:

- Stop piping the individual fields through SLURM and CLI.
- Only ever pass a path to a fully built config.

***

## 4. Offload Orchestration to a Workflow Engine (Optional, but Powerful)

For multi‑stage pipelines (preprocessing → self‑supervised → supervised → distillation → evaluation) your current model of “manually linking scripts via CLI flags” is workable but scales poorly in complexity.

In HPC contexts, a common next step is a workflow engine that *natively* talks to SLURM:

- **Snakemake**: Very popular in science; has first‑class SLURM executors and profiles, and can submit itself as a single SLURM job that then spawns cluster jobs for individual rules.[^8][^9][^6]
- **Luigi / Airflow / Metaflow**: More generic pipelines; some labs use these to coordinate SLURM jobs.
- **Custom Python orchestrator**: For smaller teams, a bespoke `pipeline.py` that creates config files and submits arrays is often enough.

With Snakemake+SLURM, for example, you describe *logical* steps (e.g. “train_uncurriculum”, “train_curriculum”, “evaluate”) in a `Snakefile`. Snakemake then handles:

- Scheduling steps in the right order.
- Submitting each training or eval job to SLURM with the requested resources.
- Passing config and paths as file dependencies, not via opaque CLI flags.[^9][^8][^6]

This lets you think at the level of “data and artifacts flowing through a DAG” rather than “what flags are on this sbatch script?”. It is increasingly used for ML on HPC clusters.[^10][^8][^6]

***

## 5. Logging and Artifacts: Run IDs over Path Encodings

Another fragility in the current design is “the folder path encodes everything about the run”. This is intuitive at first, but over time:

- Paths become unreadable (`set_02_curriculum_bs7782_s0.6_kdDISABLED/...`).
- Renaming a concept (e.g. `kd` → `distill`) requires large-scale path surgery.
- Debugging which code version or seed produced a run is tedious.

Best practice in HPC ML setups is:

- Use *simple*, predictable directory schemes keyed by run IDs and a short human label.
- Store all detailed metadata in:
    - the frozen `config.yaml` in that directory, and
    - a small `manifest.json` with key metrics and artifact locations.

This matches patterns from HPC/ML best-practices guides, which recommend using the SLURM job ID or a run ID in directory names to avoid clobbering and to reduce the “blast radius” of mistakes.[^11][^4]

For example:

```text
runs/automotive/hcrl_ch/2026-01-26_220101_supervised_curriculum/
  config.yaml
  manifest.json      # seed, git hash, final metrics, artifact paths
  checkpoints/
  logs/
  tensorboard/
```

Downstream analysis scripts then read `manifest.json` instead of re-parsing path names or SLURM logs.

***

## 6. Concrete Migration Plan from Your Current Setup

Given your existing hierarchy and the fact you already invested a lot, the goal is to *wrap* what you have rather than rewrite from scratch. A pragmatic plan:

1. **Introduce a “run” directory and frozen config:**
    - Write a small `submit_run.py` that:
        - Takes a logical experiment identifier (`dataset`, `task`, maybe `kd_status`).
        - Composes or copies a config into `runs/.../config.yaml`.
        - Writes a minimal `sbatch` command to submit a generic script with `CONFIG_PATH`.
    - Keep your existing training scripts, but add a “load config from file” path and gradually deprecate passing in individual flags.
2. **Gradually consolidate CLI flags into config fields:**
    - For each new experiment, stop using CLI options for new tuning parameters.
    - Move validation and defaulting logic into a single config layer (Pydantic v2 / Hydra structured configs).
    - When you touch a training script, remove old CLI args in favor of reading from the config object.
3. **Replace per-job SLURM scripts with an array+manifest:**
    - Encode your current CLI grid (datasets × seeds × BS settings × models) into a CSV manifest.
    - Write one array sbatch that maps `SLURM_ARRAY_TASK_ID` → row → call `submit_run.py`.
    - Once that works, delete the N special‑case scripts under `slurm_runs/`.
4. **Add a tiny run manifest for artifacts:**
    - At the end of each training run, have your script write `manifest.json` containing:
        - Config hash or full config snapshot.
        - Metrics of interest (best val, final test, etc.).
        - Paths to checkpoints and logs.
        - SLURM job ID, git SHA.
    - Point your analysis notebooks at these manifests rather than scanning folders manually.
5. **Optional: adopt a workflow engine for multi-stage pipelines:**
    - If your RL + curriculum + distillation pipeline gets more complex, consider encoding it in Snakemake and using its SLURM executor.[^8][^6][^9]
    - Start very small: one rule = one of your existing Python scripts, with the same configs.

***

## 7. Why This Will Feel Much Less Fragile

Once you adopt “frozen config + manifest + arrays” as the core pattern, your pipeline tends to:

- Become **reproducible**: any run is `python train.py --config runs/.../config.yaml`.
- Become **inspectable**: no need to infer hyperparameters from path names or logs.
- Reduce **surface area for breakage**: you rarely touch SLURM scripts; experiments evolve via config files.
- Improve **scalability**: arrays + manifest scale to hundreds or thousands of runs without generating thousands of bash scripts.[^5][^3][^4]

And importantly for a solo PhD workflow, you still retain the low-friction, single‑repo, single‑cluster style you already have—just with sharper boundaries between:

- “What experiment is this?” → config file.
- “How does it run on the cluster?” → a small, mostly static sbatch template or Snakemake profile.
- “Where are the artifacts?” → run directory + manifest.

If you describe one of your existing training entrypoints and current CLI/API, a next step would be to sketch a concrete refactor of that file into the frozen-config style so you can see exactly what would change.
<span style="display:none">[^12][^13][^14][^15][^16][^17]</span>

<div align="center">⁂</div>

[^1]: https://hydra.cc/docs/patterns/write_protect_config_node/

[^2]: https://xlstm-jax.readthedocs.io/en/latest/configuration_with_hydra.html

[^3]: https://docs.rc.asu.edu/slurm-job-array-examples/

[^4]: https://jamesthorne.com/hpc-old/

[^5]: https://stackoverflow.com/questions/69951360/in-slurm-how-to-submit-multiple-experimental-runs-in-a-batch-and-excute-them-c

[^6]: https://snakemake.readthedocs.io/en/v7.29.0/executing/cluster.html

[^7]: https://stackoverflow.com/questions/78694265/how-to-avoid-repeating-blocks-in-hydra-configs

[^8]: https://bihealth.github.io/bih-cluster/slurm/snakemake/

[^9]: https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html

[^10]: https://danmackinlay.name/notebook/hpc_hell_jobs.html

[^11]: https://info.gwdg.de/news/best-practices-for-machine-learning-with-hpc/

[^12]: image.jpg

[^13]: image.jpg

[^14]: https://slurm.schedmd.com/slurm.conf.html

[^15]: https://www.whitefiber.com/blog/understanding-slurm-for-ai-ml-workloads

[^16]: https://gitlab.com/mlops3701614/mlops_study_task_4_hydra

[^17]: https://github.com/sara-nl/hydra-2.0/blob/main/NEWS.md




<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# This is a typical CLI call I make to run a pipeline, but I think it's outdated now: ./can-train pipeline --model vgae,gat,dqn --learning-type unsupervised,supervised,rl_fusion --mode autoencoder,curriculum,fusion --dataset hcrl_sa --distillation with-kd --submit

The call you pasted is exactly the kind of “do everything via one mega‑CLI” pattern that becomes painful over time:

```bash
./can-train pipeline \
  --model vgae,gat,dqn \
  --learning-type unsupervised,supervised,rl_fusion \
  --mode autoencoder,curriculum,fusion \
  --dataset hcrl_sa \
  --distillation with-kd \
  --submit
```

It’s doing three jobs at once:

1. Defining a *grid* of experiments (3 models × 3 learning types × 3 modes).
2. Describing *what each run should do* (dataset, distillation, etc.).
3. Controlling *how to run it* (submit to SLURM).

Below is how to modernize this into the “frozen config + manifest + simple CLI” pattern from the previous message.

***

## 1. Turn that CLI into a pipeline config file

Instead of encoding everything in arguments, encode the pipeline in a YAML/JSON config. For example:

`configs/pipelines/hcrl_sa_with_kd.yaml`:

```yaml
dataset: hcrl_sa
distillation: with_kd

stages:
  - name: vgae_autoencoder_unsupervised
    model: vgae
    learning_type: unsupervised
    mode: autoencoder

  - name: gat_curriculum_supervised
    model: gat
    learning_type: supervised
    mode: curriculum

  - name: dqn_fusion_rl
    model: dqn
    learning_type: rl_fusion
    mode: fusion
```

Or if you really want a Cartesian product:

```yaml
dataset: hcrl_sa
distillation: with_kd

grid:
  model: [vgae, gat, dqn]
  learning_type: [unsupervised, supervised, rl_fusion]
  mode: [autoencoder, curriculum, fusion]
```

Your Python “launcher” then:

- Reads this pipeline config.
- Either:
    - expands the `grid` into concrete runs, or
    - iterates explicit `stages` as listed.
- For each run, writes a *frozen* per‑run config into `runs/.../config.yaml`.
- Submits a generic SLURM job that calls `train.py --config runs/.../config.yaml`.

***

## 2. Replace the mega‑CLI with a tiny, stable interface

Once the pipeline is encoded in config, your day‑to‑day CLI can be as simple as:

```bash
# Submit whole pipeline
can submit-pipeline --config configs/pipelines/hcrl_sa_with_kd.yaml

# Or, if you like explicit subcommands:
can pipeline submit --config configs/pipelines/hcrl_sa_with_kd.yaml
```

Internally, `submit-pipeline` does the work your old `pipeline` subcommand did:

- Expands the grid / stages.
- Creates run directories.
- Writes per‑run configs.
- Submits SLURM jobs (possibly as an array).

No individual hyperparameters need to go through the CLI anymore; the CLI just points to *one* config file.

***

## 3. Make per‑run configs explicit artifacts

For each expanded run, you end up with something like:

`runs/automotive/hcrl_sa/2026-01-26_220101_vgae_autoencoder_unsupervised/config.yaml`:

```yaml
dataset: hcrl_sa
distillation: with_kd
model: vgae
learning_type: unsupervised
mode: autoencoder

# plus all the usual training / optimizer / scheduler / paths, etc.
```

SLURM script (static template) only cares about:

```bash
python train.py --config "$CONFIG_PATH"
```

And `train.py` never sees `--model` or `--learning-type` as CLI flags; it just deserializes the config.

***

## 4. Optionally, drive this via a manifest + job array

If you like the idea of a CSV manifest, your `can submit-pipeline` command can produce a manifest instead of submitting each job directly.

Example `pipelines/hcrl_sa_with_kd_manifest.csv` (auto‑generated from the grid):

```csv
id,run_name,config_path
0,vgae_autoencoder_unsupervised,runs/.../config_0.yaml
1,vgae_curriculum_supervised,runs/.../config_1.yaml
...
```

Then one generic SLURM array job reads that CSV and calls:

```bash
python train.py --config "$(config_path_for $SLURM_ARRAY_TASK_ID)"
```

You’ve completely eliminated the “explode CLI into N job‑specific scripts” step.

***

## 5. Mapping old flags to new structure

Conceptually:


| Old CLI piece | New home |
| :-- | :-- |
| `--model ...` | `model` field in per‑run `config.yaml` |
| `--learning-type ...` | `learning_type` field in per‑run config |
| `--mode ...` | `mode` field in per‑run config |
| `--dataset hcrl_sa` | top‑level `dataset` in pipeline + run config |
| `--distillation with-kd` | `distillation` in pipeline + run config |
| `--submit` | separate command (`submit-pipeline`) or flag |

The big win is: instead of parsing and expanding comma‑separated strings in a single CLI, you push that expressiveness into a config file (or grid specification), and let a small, testable Python module turn that into concrete run configs and SLURM submissions.

If you’d like, next step could be to take the real argument parser for `can-train pipeline` and sketch an exact refactor into a `PipelineConfig` dataclass / Pydantic model plus a `submit_pipeline(config: PipelineConfig)` function, so you have a drop‑in migration path rather than a rewrite.
<span style="display:none">[^1][^2]</span>

<div align="center">⁂</div>

[^1]: image.jpg

[^2]: image.jpg

