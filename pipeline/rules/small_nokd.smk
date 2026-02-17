# Small without KD (ablation -- no teacher dependency).

rule vgae_small_nokd:
    input:
        cache=rules.preprocess.output[0],
    output:
        done=_done("{ds}", "vgae", "small", "autoencoder"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "vgae", "small", "autoencoder"),
        err=_log("{ds}", "vgae", "small", "autoencoder", stream="err"),
    shell:
        CLI + " autoencoder --model vgae --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule gat_small_nokd:
    input:
        vgae=_done("{ds}", "vgae", "small", "autoencoder"),
    output:
        done=_done("{ds}", "gat", "small", "curriculum"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "gat", "small", "curriculum"),
        err=_log("{ds}", "gat", "small", "curriculum", stream="err"),
    shell:
        CLI + " curriculum --model gat --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule dqn_small_nokd:
    input:
        vgae=_done("{ds}", "vgae", "small", "autoencoder"),
        gat=_done("{ds}", "gat", "small", "curriculum"),
    output:
        done=_done("{ds}", "dqn", "small", "fusion"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "dqn", "small", "fusion"),
        err=_log("{ds}", "dqn", "small", "fusion", stream="err"),
    shell:
        CLI + " fusion --model dqn --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"
