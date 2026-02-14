# Reports: parameterized notebooks via Papermill.

rule notebook_report:
    input:
        _metrics("{ds}", "vgae", "large", "evaluation"),
    output:
        EXPERIMENT_ROOT + "/{ds}/report/analysis.ipynb",
    resources:
        time_min=30, mem_mb=8000, cpus_per_task=4, gpus=0,
        slurm_account=SLURM_ACCOUNT, slurm_partition="serial",
    shell:
        PY + " -m papermill notebooks/03_analytics.ipynb {output} -p dataset {wildcards.ds}"
