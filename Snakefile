"""
Snakemake Pipeline for KD-GAT Graph Neural Network Training

This replaces the custom can-train CLI + job_manager.py system with a
declarative, file-based pipeline that automatically handles dependencies
and SLURM job submission.

Pipeline Structure:
    VGAE (autoencoder) → GAT (curriculum) → DQN (fusion) → Evaluation

Usage:
    # Dry run (see what will be executed)
    snakemake --profile profiles/slurm -n

    # Run all pipelines
    snakemake --profile profiles/slurm --jobs 20

    # Run specific dataset
    snakemake --profile profiles/slurm results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json

    # Resume from failures
    snakemake --profile profiles/slurm --jobs 20 --rerun-incomplete

    # Generate pipeline DAG visualization
    snakemake --dag | dot -Tpdf > pipeline_dag.pdf
"""

# ============================================================================
# Configuration
# ============================================================================

configfile: "config/snakemake_config.yaml"

# Get configuration with defaults
DATASETS = config.get("datasets", ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"])
MODALITIES = config.get("modalities", ["automotive"])
MODEL_SIZES = config.get("model_sizes", ["teacher", "student"])
DISTILLATION_MODES = config.get("distillation_modes", {"teacher": ["no_distillation"], "student": ["with_kd"]})

# SLURM defaults (can be overridden in profile)
SLURM_ACCOUNT = config.get("slurm_account", "PAS3209")
SLURM_PARTITION = config.get("slurm_partition", "gpu")
SLURM_EMAIL = config.get("slurm_email", "frugoli.1@osu.edu")

# ============================================================================
# Helper Functions
# ============================================================================

def get_experiment_dir(modality, dataset, model, model_size, distillation):
    """Generate canonical experiment directory path."""
    learning_type_map = {
        "vgae": "unsupervised",
        "gat": "supervised",
        "dqn": "rl_fusion"
    }
    mode_map = {
        "vgae": "autoencoder",
        "gat": "curriculum",
        "dqn": "fusion"
    }

    learning_type = learning_type_map[model]
    mode = mode_map[model]

    return f"experimentruns/{modality}/{dataset}/{learning_type}/{model}/{model_size}/{distillation}/{mode}"

def get_model_path(modality, dataset, model, model_size, distillation):
    """Get path to trained model checkpoint."""
    exp_dir = get_experiment_dir(modality, dataset, model, model_size, distillation)
    # Use the best model checkpoint
    return f"{exp_dir}/models/{model}_{model_size}_{distillation}_best.pth"

def get_teacher_path(modality, dataset, model):
    """Get path to teacher model for distillation."""
    return get_model_path(modality, dataset, model, "teacher", "no_distillation")

# ============================================================================
# Target Rules (define what to build)
# ============================================================================

rule all:
    """Default target: train and evaluate all pipelines."""
    input:
        # Teacher pipelines (no distillation)
        expand(
            "results/{modality}/{dataset}/teacher/no_distillation/evaluation/{model}_eval.json",
            modality=MODALITIES,
            dataset=DATASETS,
            model=["vgae", "gat", "dqn"]
        ),
        # Student pipelines (with knowledge distillation)
        expand(
            "results/{modality}/{dataset}/student/with_kd/evaluation/{model}_eval.json",
            modality=MODALITIES,
            dataset=DATASETS,
            model=["vgae", "gat", "dqn"]
        )

rule all_teachers:
    """Train and evaluate only teacher models."""
    input:
        expand(
            "results/{modality}/{dataset}/teacher/no_distillation/evaluation/{model}_eval.json",
            modality=MODALITIES,
            dataset=DATASETS,
            model=["vgae", "gat", "dqn"]
        )

rule all_students:
    """Train and evaluate only student models (requires teachers)."""
    input:
        expand(
            "results/{modality}/{dataset}/student/with_kd/evaluation/{model}_eval.json",
            modality=MODALITIES,
            dataset=DATASETS,
            model=["vgae", "gat", "dqn"]
        )

# ============================================================================
# Training Rules (VGAE → GAT → DQN)
# ============================================================================

rule train_vgae:
    """Train VGAE autoencoder (stage 1 of pipeline)."""
    output:
        model=get_model_path("{modality}", "{dataset}", "vgae", "{model_size}", "{distillation}"),
        checkpoint=get_experiment_dir("{modality}", "{dataset}", "vgae", "{model_size}", "{distillation}") + "/checkpoints/last.ckpt"
    params:
        exp_dir=lambda w: get_experiment_dir(w.modality, w.dataset, "vgae", w.model_size, w.distillation),
        teacher_flag=lambda w: f"--teacher-path {get_teacher_path(w.modality, w.dataset, 'vgae')}" if w.distillation == "with_kd" else ""
    log:
        out=get_experiment_dir("{modality}", "{dataset}", "vgae", "{model_size}", "{distillation}") + "/logs/training.log",
        err=get_experiment_dir("{modality}", "{dataset}", "vgae", "{model_size}", "{distillation}") + "/logs/training.err"
    resources:
        partition=SLURM_PARTITION,
        account=SLURM_ACCOUNT,
        time_min=360,
        mem_mb=64000,
        cpus_per_task=16,
        gpus=1,
        gpu_type="v100"
    conda:
        "envs/gnn-experiments.yaml"
    shell:
        """
        mkdir -p {params.exp_dir}/models {params.exp_dir}/checkpoints {params.exp_dir}/logs

        python train_with_hydra_zen.py \
            --model vgae \
            --model-size {wildcards.model_size} \
            --dataset {wildcards.dataset} \
            --modality {wildcards.modality} \
            --training autoencoder \
            --learning-type unsupervised \
            {params.teacher_flag} \
            --output-dir {params.exp_dir} \
            > {log.out} 2> {log.err}
        """

rule train_gat:
    """Train GAT classifier with curriculum learning (stage 2 - depends on VGAE)."""
    input:
        vgae_model=lambda w: get_model_path(w.modality, w.dataset, "vgae", w.model_size, w.distillation)
    output:
        model=get_model_path("{modality}", "{dataset}", "gat", "{model_size}", "{distillation}"),
        checkpoint=get_experiment_dir("{modality}", "{dataset}", "gat", "{model_size}", "{distillation}") + "/checkpoints/last.ckpt"
    params:
        exp_dir=lambda w: get_experiment_dir(w.modality, w.dataset, "gat", w.model_size, w.distillation),
        teacher_flag=lambda w: f"--teacher-path {get_teacher_path(w.modality, w.dataset, 'gat')}" if w.distillation == "with_kd" else ""
    log:
        out=get_experiment_dir("{modality}", "{dataset}", "gat", "{model_size}", "{distillation}") + "/logs/training.log",
        err=get_experiment_dir("{modality}", "{dataset}", "gat", "{model_size}", "{distillation}") + "/logs/training.err"
    resources:
        partition=SLURM_PARTITION,
        account=SLURM_ACCOUNT,
        time_min=360,
        mem_mb=64000,
        cpus_per_task=16,
        gpus=1,
        gpu_type="v100"
    conda:
        "envs/gnn-experiments.yaml"
    shell:
        """
        mkdir -p {params.exp_dir}/models {params.exp_dir}/checkpoints {params.exp_dir}/logs

        python train_with_hydra_zen.py \
            --model gat \
            --model-size {wildcards.model_size} \
            --dataset {wildcards.dataset} \
            --modality {wildcards.modality} \
            --training curriculum \
            --learning-type supervised \
            {params.teacher_flag} \
            --output-dir {params.exp_dir} \
            > {log.out} 2> {log.err}
        """

rule train_dqn:
    """Train DQN fusion agent (stage 3 - depends on VGAE + GAT)."""
    input:
        vgae_model=lambda w: get_model_path(w.modality, w.dataset, "vgae", w.model_size, w.distillation),
        gat_model=lambda w: get_model_path(w.modality, w.dataset, "gat", w.model_size, w.distillation)
    output:
        model=get_model_path("{modality}", "{dataset}", "dqn", "{model_size}", "{distillation}"),
        checkpoint=get_experiment_dir("{modality}", "{dataset}", "dqn", "{model_size}", "{distillation}") + "/checkpoints/last.ckpt"
    params:
        exp_dir=lambda w: get_experiment_dir(w.modality, w.dataset, "dqn", w.model_size, w.distillation),
        teacher_flag=lambda w: f"--teacher-path {get_teacher_path(w.modality, w.dataset, 'dqn')}" if w.distillation == "with_kd" else ""
    log:
        out=get_experiment_dir("{modality}", "{dataset}", "dqn", "{model_size}", "{distillation}") + "/logs/training.log",
        err=get_experiment_dir("{modality}", "{dataset}", "dqn", "{model_size}", "{distillation}") + "/logs/training.err"
    resources:
        partition=SLURM_PARTITION,
        account=SLURM_ACCOUNT,
        time_min=360,
        mem_mb=64000,
        cpus_per_task=16,
        gpus=1,
        gpu_type="v100"
    conda:
        "envs/gnn-experiments.yaml"
    shell:
        """
        mkdir -p {params.exp_dir}/models {params.exp_dir}/checkpoints {params.exp_dir}/logs

        python train_with_hydra_zen.py \
            --model dqn \
            --model-size {wildcards.model_size} \
            --dataset {wildcards.dataset} \
            --modality {wildcards.modality} \
            --training fusion \
            --learning-type rl_fusion \
            --vgae-path {input.vgae_model} \
            --gat-path {input.gat_model} \
            {params.teacher_flag} \
            --output-dir {params.exp_dir} \
            > {log.out} 2> {log.err}
        """

# ============================================================================
# Evaluation Rules
# ============================================================================

rule evaluate_model:
    """Evaluate trained model and generate metrics."""
    input:
        model=get_model_path("{modality}", "{dataset}", "{model}", "{model_size}", "{distillation}"),
        # For DQN, also need VGAE and GAT models
        vgae_model=lambda w: get_model_path(w.modality, w.dataset, "vgae", w.model_size, w.distillation) if w.model == "dqn" else [],
        gat_model=lambda w: get_model_path(w.modality, w.dataset, "gat", w.model_size, w.distillation) if w.model == "dqn" else []
    output:
        json="results/{modality}/{dataset}/{model_size}/{distillation}/evaluation/{model}_eval.json",
        csv="results/{modality}/{dataset}/{model_size}/{distillation}/evaluation/{model}_eval.csv"
    params:
        mode_map=lambda w: {"vgae": "autoencoder", "gat": "curriculum", "dqn": "fusion"}[w.model],
        fusion_flags=lambda w, input: f"--vgae-path {input.vgae_model} --gat-path {input.gat_model}" if w.model == "dqn" else ""
    log:
        out="results/{modality}/{dataset}/{model_size}/{distillation}/evaluation/{model}_eval.log",
        err="results/{modality}/{dataset}/{model_size}/{distillation}/evaluation/{model}_eval.err"
    resources:
        partition=SLURM_PARTITION,
        account=SLURM_ACCOUNT,
        time_min=60,
        mem_mb=32000,
        cpus_per_task=8,
        gpus=1,
        gpu_type="v100"
    conda:
        "envs/gnn-experiments.yaml"
    shell:
        """
        mkdir -p $(dirname {output.json})

        python -m src.evaluation.evaluation \
            --dataset {wildcards.dataset} \
            --model-path {input.model} \
            --training-mode {params.mode_map} \
            --mode standard \
            --batch-size 512 \
            --device cuda \
            --csv-output {output.csv} \
            --json-output {output.json} \
            --threshold-optimization true \
            {params.fusion_flags} \
            > {log.out} 2> {log.err}
        """

# ============================================================================
# Utility Rules
# ============================================================================

rule clean_checkpoints:
    """Remove intermediate checkpoints (keep only best models)."""
    shell:
        """
        find experimentruns -type f -name "epoch*.ckpt" -delete
        find experimentruns -type f -name "last.ckpt" -delete
        echo "Cleaned intermediate checkpoints (kept best models)"
        """

rule clean_logs:
    """Remove old log files."""
    shell:
        """
        find experimentruns -type d -name "logs" -exec rm -rf {} + 2>/dev/null || true
        find results -type f -name "*.log" -delete 2>/dev/null || true
        echo "Cleaned log files"
        """

rule clean_all:
    """Remove all generated files (WARNING: deletes all experiments!)."""
    shell:
        """
        rm -rf experimentruns results
        echo "WARNING: Deleted all experimental results and models!"
        """

# ============================================================================
# Reporting Rules
# ============================================================================

rule generate_report:
    """Generate summary report of all evaluations."""
    input:
        expand(
            "results/{modality}/{dataset}/{model_size}/{distillation}/evaluation/{model}_eval.json",
            modality=MODALITIES,
            dataset=DATASETS,
            model_size=["teacher", "student"],
            distillation=["no_distillation", "with_kd"],
            model=["vgae", "gat", "dqn"]
        )
    output:
        "results/summary_report.html"
    conda:
        "envs/gnn-experiments.yaml"
    script:
        "scripts/generate_summary_report.py"

# ============================================================================
# Constraints for Student Models
# ============================================================================

# Student models with distillation must wait for corresponding teacher
ruleorder: train_vgae > train_gat > train_dqn

# Ensure student training doesn't start until teacher is complete
def get_student_dependencies(wildcards):
    """For student models, add dependency on teacher completion."""
    if wildcards.model_size == "student" and wildcards.distillation == "with_kd":
        # Student needs the teacher model to exist
        return get_teacher_path(wildcards.modality, wildcards.dataset, wildcards.model)
    return []
