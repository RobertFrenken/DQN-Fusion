# Small without KD (ablation -- no teacher dependency).

rule vgae_small_nokd:
    input:
        cache=rules.preprocess.output[0],
    output:
        _ckpt("{ds}", "vgae", "small", "autoencoder"),
    benchmark:
        _bench("{ds}", "vgae", "small", "autoencoder")
    resources: **_TRAIN_RES,
    retries: 2
    log:
        out=_log("{ds}", "vgae", "small", "autoencoder"),
        err=_log("{ds}", "vgae", "small", "autoencoder", stream="err"),
    shell:
        CLI + " autoencoder --model vgae --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"

rule gat_small_nokd:
    input:
        vgae=_ckpt("{ds}", "vgae", "small", "autoencoder"),
    output:
        _ckpt("{ds}", "gat", "small", "curriculum"),
    benchmark:
        _bench("{ds}", "gat", "small", "curriculum")
    resources: **_TRAIN_RES,
    retries: 2
    log:
        out=_log("{ds}", "gat", "small", "curriculum"),
        err=_log("{ds}", "gat", "small", "curriculum", stream="err"),
    shell:
        CLI + " curriculum --model gat --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"

rule dqn_small_nokd:
    input:
        vgae=_ckpt("{ds}", "vgae", "small", "autoencoder"),
        gat=_ckpt("{ds}", "gat", "small", "curriculum"),
    output:
        _ckpt("{ds}", "dqn", "small", "fusion"),
    benchmark:
        _bench("{ds}", "dqn", "small", "fusion")
    resources: **_TRAIN_RES,
    retries: 2
    log:
        out=_log("{ds}", "dqn", "small", "fusion"),
        err=_log("{ds}", "dqn", "small", "fusion", stream="err"),
    shell:
        CLI + " fusion --model dqn --scale small --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
