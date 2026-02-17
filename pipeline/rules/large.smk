# Large pipeline (no KD dependencies, runs first).

rule vgae_large:
    input:
        cache=rules.preprocess.output[0],
    output:
        done=_done("{ds}", "vgae", "large", "autoencoder"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "vgae", "large", "autoencoder"),
        err=_log("{ds}", "vgae", "large", "autoencoder", stream="err"),
    shell:
        CLI + " autoencoder --model vgae --scale large --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule gat_large:
    input:
        vgae=_done("{ds}", "vgae", "large", "autoencoder"),
    output:
        done=_done("{ds}", "gat", "large", "curriculum"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "gat", "large", "curriculum"),
        err=_log("{ds}", "gat", "large", "curriculum", stream="err"),
    shell:
        CLI + " curriculum --model gat --scale large --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule dqn_large:
    input:
        vgae=_done("{ds}", "vgae", "large", "autoencoder"),
        gat=_done("{ds}", "gat", "large", "curriculum"),
    output:
        done=_done("{ds}", "dqn", "large", "fusion"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "dqn", "large", "fusion"),
        err=_log("{ds}", "dqn", "large", "fusion", stream="err"),
    shell:
        CLI + " fusion --model dqn --scale large --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"
