# Preprocessing: deterministic graph cache, shared by all training rules.
# Builds both train/val cache (processed_graphs.pt) and per-scenario test
# caches (test_*.pt) so evaluation doesn't rebuild graphs from CSVs.

rule preprocess:
    output:
        directory("data/cache/{ds}"),
    resources:
        time_min=240, mem_mb=64000, cpus_per_task=8, gpus=0,
        slurm_account=SLURM_ACCOUNT, slurm_partition="cpu",
    cache: True
    shell:
        PY + " -c \""
        "from src.training.datamodules import load_dataset, load_test_scenarios; from pathlib import Path; "
        "ds_path = Path('data/automotive/{wildcards.ds}'); "
        "cache_path = Path('data/cache/{wildcards.ds}'); "
        "load_dataset('{wildcards.ds}', ds_path, cache_path); "
        "load_test_scenarios('{wildcards.ds}', ds_path, cache_path)"
        "\""
