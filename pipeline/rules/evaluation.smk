# Evaluation rules (one per pipeline variant, grouped for SLURM efficiency).

rule eval_large:
    input:
        vgae=_ckpt("{ds}", "vgae", "large", "autoencoder"),
        gat=_ckpt("{ds}", "gat", "large", "curriculum"),
        dqn=_ckpt("{ds}", "dqn", "large", "fusion"),
    group: "evaluation"
    output:
        report(
            _metrics("{ds}", "vgae", "large", "evaluation"),
            category="Evaluation",
            caption="Large model evaluation metrics for dataset {ds}",
        ),
    benchmark:
        _bench("{ds}", "vgae", "large", "evaluation")
    resources: **_EVAL_RES,
    log:
        out=_log("{ds}", "vgae", "large", "evaluation"),
        err=_log("{ds}", "vgae", "large", "evaluation", stream="err"),
    shell:
        CLI + " evaluation --model vgae --scale large --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"

rule eval_small_kd:
    input:
        vgae=_ckpt("{ds}", "vgae", "small", "autoencoder", aux="kd"),
        gat=_ckpt("{ds}", "gat", "small", "curriculum", aux="kd"),
        dqn=_ckpt("{ds}", "dqn", "small", "fusion", aux="kd"),
    group: "evaluation"
    output:
        report(
            _metrics("{ds}", "vgae", "small", "evaluation", aux="kd"),
            category="Evaluation",
            caption="Small+KD evaluation metrics for dataset {ds}",
        ),
    benchmark:
        _bench("{ds}", "vgae", "small", "evaluation", aux="kd")
    resources: **_EVAL_RES,
    log:
        out=_log("{ds}", "vgae", "small", "evaluation", aux="kd"),
        err=_log("{ds}", "vgae", "small", "evaluation", aux="kd", stream="err"),
    shell:
        CLI + " evaluation --model vgae --scale small --auxiliaries kd_standard --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"

rule eval_small_nokd:
    input:
        vgae=_ckpt("{ds}", "vgae", "small", "autoencoder"),
        gat=_ckpt("{ds}", "gat", "small", "curriculum"),
        dqn=_ckpt("{ds}", "dqn", "small", "fusion"),
    group: "evaluation"
    output:
        report(
            _metrics("{ds}", "vgae", "small", "evaluation"),
            category="Evaluation",
            caption="Small (no KD) evaluation metrics for dataset {ds}",
        ),
    benchmark:
        _bench("{ds}", "vgae", "small", "evaluation")
    resources: **_EVAL_RES,
    log:
        out=_log("{ds}", "vgae", "small", "evaluation"),
        err=_log("{ds}", "vgae", "small", "evaluation", stream="err"),
    shell:
        CLI + " evaluation --model vgae --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
