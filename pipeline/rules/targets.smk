# Target rules (meta-rules that pull in pipeline variants).

rule all:
    input:
        # Large pipeline
        expand(_done("{ds}", "dqn", "large", "fusion"), ds=DATASETS),
        # Small with KD
        expand(_done("{ds}", "dqn", "small", "fusion", aux="kd"), ds=DATASETS),
        # Small without KD (ablation)
        expand(_done("{ds}", "dqn", "small", "fusion"), ds=DATASETS),
        # Evaluation (all variants)
        expand(_done("{ds}", "eval", "large", "evaluation"), ds=DATASETS),
        expand(_done("{ds}", "eval", "small", "evaluation", aux="kd"), ds=DATASETS),
        expand(_done("{ds}", "eval", "small", "evaluation"), ds=DATASETS),

rule large:
    input:
        expand(_done("{ds}", "dqn", "large", "fusion"), ds=DATASETS),

rule small_kd:
    input:
        expand(_done("{ds}", "dqn", "small", "fusion", aux="kd"), ds=DATASETS),

rule small_nokd:
    input:
        expand(_done("{ds}", "dqn", "small", "fusion"), ds=DATASETS),

rule evaluate_all:
    input:
        expand(_done("{ds}", "eval", "large", "evaluation"), ds=DATASETS),
        expand(_done("{ds}", "eval", "small", "evaluation", aux="kd"), ds=DATASETS),
        expand(_done("{ds}", "eval", "small", "evaluation"), ds=DATASETS),
