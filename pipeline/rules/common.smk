# Common aliases and SLURM resource definitions.
# All variables from the main Snakefile (DATASETS, PY, CLI, EXPERIMENT_ROOT)
# are available via Snakemake's shared include namespace.

_ckpt    = checkpoint_path_str
_metrics = metrics_path_str
_bench   = benchmark_path_str
_log     = log_path_str
_done    = done_path_str

_SLURM     = dict(slurm_account=SLURM_ACCOUNT, slurm_partition=SLURM_PARTITION, gpu_type=SLURM_GPU_TYPE)
_TRAIN_RES = dict(time_min=360, mem_mb=lambda wc, attempt: 128000 * attempt, cpus_per_task=16, gpus=1, **_SLURM)
_EVAL_RES  = dict(time_min=120, mem_mb=32000, cpus_per_task=8, gpus=1, **_SLURM)
