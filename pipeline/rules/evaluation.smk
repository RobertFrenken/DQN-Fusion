# Evaluation rules (one per pipeline variant, independent SLURM jobs).
# NOTE: group: "evaluation" was removed because it causes Snakemake to cancel
# ALL group members when one fails, turning partial failures into total failures.

rule eval_large:
    input:
        vgae=_done("{ds}", "vgae", "large", "autoencoder"),
        gat=_done("{ds}", "gat", "large", "curriculum"),
        dqn=_done("{ds}", "dqn", "large", "fusion"),
    output:
        done=_done("{ds}", "eval", "large", "evaluation"),
    resources: **_EVAL_RES,
    cache: True
    log:
        out=_log("{ds}", "eval", "large", "evaluation"),
        err=_log("{ds}", "eval", "large", "evaluation", stream="err"),
    shell:
        CLI + " evaluation --model vgae --scale large --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule eval_small_kd:
    input:
        vgae=_done("{ds}", "vgae", "small", "autoencoder", aux="kd"),
        gat=_done("{ds}", "gat", "small", "curriculum", aux="kd"),
        dqn=_done("{ds}", "dqn", "small", "fusion", aux="kd"),
    output:
        done=_done("{ds}", "eval", "small", "evaluation", aux="kd"),
    resources: **_EVAL_RES,
    cache: True
    log:
        out=_log("{ds}", "eval", "small", "evaluation", aux="kd"),
        err=_log("{ds}", "eval", "small", "evaluation", aux="kd", stream="err"),
    shell:
        CLI + " evaluation --model vgae --scale small --auxiliaries kd_standard --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule eval_small_nokd:
    input:
        vgae=_done("{ds}", "vgae", "small", "autoencoder"),
        gat=_done("{ds}", "gat", "small", "curriculum"),
        dqn=_done("{ds}", "dqn", "small", "fusion"),
    output:
        done=_done("{ds}", "eval", "small", "evaluation"),
    resources: **_EVAL_RES,
    cache: True
    log:
        out=_log("{ds}", "eval", "small", "evaluation"),
        err=_log("{ds}", "eval", "small", "evaluation", stream="err"),
    shell:
        CLI + " evaluation --model vgae --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

