# Preprocessing: deterministic graph cache, shared by all training rules.

rule preprocess:
    output:
        directory("data/cache/{ds}"),
    resources:
        time_min=120, mem_mb=64000, cpus_per_task=8, gpus=0,
        slurm_account=SLURM_ACCOUNT, slurm_partition="serial",
    cache: True
    shell:
        PY + " -c \""
        "from src.training.datamodules import load_dataset; from pathlib import Path; "
        "load_dataset('{wildcards.ds}', Path('data/automotive/{wildcards.ds}'), Path('data/cache/{wildcards.ds}'))"
        "\""
