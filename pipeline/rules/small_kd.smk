# Small with KD (each stage depends on large counterpart as teacher).

rule vgae_small_kd:
    input:
        teacher=_ckpt("{ds}", "vgae", "large", "autoencoder"),
        cache=rules.preprocess.output[0],
    output:
        _ckpt("{ds}", "vgae", "small", "autoencoder", aux="kd"),
    benchmark:
        _bench("{ds}", "vgae", "small", "autoencoder", aux="kd")
    resources: **_TRAIN_RES,
    retries: 2
    log:
        out=_log("{ds}", "vgae", "small", "autoencoder", aux="kd"),
        err=_log("{ds}", "vgae", "small", "autoencoder", aux="kd", stream="err"),
    shell:
        CLI + " autoencoder --model vgae --scale small --auxiliaries kd_standard"
        " --teacher-path {input.teacher} --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"

rule gat_small_kd:
    input:
        teacher=_ckpt("{ds}", "gat", "large", "curriculum"),
        vgae=_ckpt("{ds}", "vgae", "small", "autoencoder", aux="kd"),
    output:
        _ckpt("{ds}", "gat", "small", "curriculum", aux="kd"),
    benchmark:
        _bench("{ds}", "gat", "small", "curriculum", aux="kd")
    resources: **_TRAIN_RES,
    retries: 2
    log:
        out=_log("{ds}", "gat", "small", "curriculum", aux="kd"),
        err=_log("{ds}", "gat", "small", "curriculum", aux="kd", stream="err"),
    shell:
        CLI + " curriculum --model gat --scale small --auxiliaries kd_standard"
        " --teacher-path {input.teacher} --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"

rule dqn_small_kd:
    input:
        teacher=_ckpt("{ds}", "dqn", "large", "fusion"),
        vgae=_ckpt("{ds}", "vgae", "small", "autoencoder", aux="kd"),
        gat=_ckpt("{ds}", "gat", "small", "curriculum", aux="kd"),
    output:
        _ckpt("{ds}", "dqn", "small", "fusion", aux="kd"),
    benchmark:
        _bench("{ds}", "dqn", "small", "fusion", aux="kd")
    resources: **_TRAIN_RES,
    retries: 2
    log:
        out=_log("{ds}", "dqn", "small", "fusion", aux="kd"),
        err=_log("{ds}", "dqn", "small", "fusion", aux="kd", stream="err"),
    shell:
        CLI + " fusion --model dqn --scale small --auxiliaries kd_standard"
        " --teacher-path {input.teacher} --dataset {wildcards.ds}"
        " > {log.out} 2> {log.err}"
