# Small with KD (each stage depends on large counterpart as teacher).

rule vgae_small_kd:
    input:
        teacher=_done("{ds}", "vgae", "large", "autoencoder"),
        cache=rules.preprocess.output[0],
    output:
        done=_done("{ds}", "vgae", "small", "autoencoder", aux="kd"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "vgae", "small", "autoencoder", aux="kd"),
        err=_log("{ds}", "vgae", "small", "autoencoder", aux="kd", stream="err"),
    shell:
        CLI + " autoencoder --model vgae --scale small --auxiliaries kd_standard"
        " --teacher-path " + _ckpt("{ds}", "vgae", "large", "autoencoder")
        + " --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule gat_small_kd:
    input:
        teacher=_done("{ds}", "gat", "large", "curriculum"),
        vgae=_done("{ds}", "vgae", "small", "autoencoder", aux="kd"),
    output:
        done=_done("{ds}", "gat", "small", "curriculum", aux="kd"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "gat", "small", "curriculum", aux="kd"),
        err=_log("{ds}", "gat", "small", "curriculum", aux="kd", stream="err"),
    shell:
        CLI + " curriculum --model gat --scale small --auxiliaries kd_standard"
        " --teacher-path " + _ckpt("{ds}", "gat", "large", "curriculum")
        + " --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"

rule dqn_small_kd:
    input:
        teacher=_done("{ds}", "dqn", "large", "fusion"),
        vgae=_done("{ds}", "vgae", "small", "autoencoder", aux="kd"),
        gat=_done("{ds}", "gat", "small", "curriculum", aux="kd"),
    output:
        done=_done("{ds}", "dqn", "small", "fusion", aux="kd"),
    resources: **_TRAIN_RES,
    retries: 2
    cache: True
    log:
        out=_log("{ds}", "dqn", "small", "fusion", aux="kd"),
        err=_log("{ds}", "dqn", "small", "fusion", aux="kd", stream="err"),
    shell:
        CLI + " fusion --model dqn --scale small --auxiliaries kd_standard"
        " --teacher-path " + _ckpt("{ds}", "dqn", "large", "fusion")
        + " --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
        " && touch {output.done}"
