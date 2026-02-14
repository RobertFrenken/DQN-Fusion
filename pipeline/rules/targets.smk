# Target rules (meta-rules that pull in pipeline variants).

rule all:
    input:
        # Large pipeline
        expand(_ckpt("{ds}", "dqn", "large", "fusion"), ds=DATASETS),
        # Small with KD
        expand(_ckpt("{ds}", "dqn", "small", "fusion", aux="kd"), ds=DATASETS),
        # Small without KD (ablation)
        expand(_ckpt("{ds}", "dqn", "small", "fusion"), ds=DATASETS),

rule large:
    input:
        expand(_ckpt("{ds}", "dqn", "large", "fusion"), ds=DATASETS),

rule small_kd:
    input:
        expand(_ckpt("{ds}", "dqn", "small", "fusion", aux="kd"), ds=DATASETS),

rule small_nokd:
    input:
        expand(_ckpt("{ds}", "dqn", "small", "fusion"), ds=DATASETS),

rule evaluate_all:
    input:
        expand(_metrics("{ds}", "vgae", "large", "evaluation"), ds=DATASETS),
        expand(_metrics("{ds}", "vgae", "small", "evaluation", aux="kd"), ds=DATASETS),
        expand(_metrics("{ds}", "vgae", "small", "evaluation"), ds=DATASETS),
