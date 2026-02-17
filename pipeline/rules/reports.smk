# Reports: parameterized notebooks via Papermill + pipeline visualization.

rule rulegraph:
    output:
        "docs/pipeline_dag.svg",
    localrule: True
    shell:
        "PYTHONPATH=. snakemake -s pipeline/Snakefile --rulegraph all | dot -Tsvg > {output}"

rule notebook_report:
    input:
        _done("{ds}", "eval", "large", "evaluation"),
    output:
        EXPERIMENT_ROOT + "/{ds}/report/analysis.ipynb",
    resources:
        time_min=30, mem_mb=8000, cpus_per_task=4, gpus=0,
        slurm_account=SLURM_ACCOUNT, slurm_partition="cpu",
    shell:
        PY + " -m papermill notebooks/03_analytics.ipynb {output} -p dataset {wildcards.ds}"
