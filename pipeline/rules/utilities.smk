# Utility rules: cleanup helpers.

rule clean_logs:
    shell: "find {EXPERIMENT_ROOT} -name 'slurm.*' -delete"

rule clean_all:
    shell: "rm -rf {EXPERIMENT_ROOT}"
