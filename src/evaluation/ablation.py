"""
Ablation Study Framework for CAN-Graph Models

Compares multiple trained models to isolate the impact of specific design choices:
- Knowledge Distillation Impact (KD vs No-KD)
- Curriculum Learning Impact (Curriculum vs Standard)
- Fusion Strategy Impact (DQN vs Static)
- Training Mode Impact (Normal vs Autoencoder vs Curriculum)
"""

import argparse
import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

from src.evaluation.evaluation import Evaluator, EvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model in an ablation study."""
    name: str
    dataset: str
    model_path: str
    teacher_path: str = None
    training_mode: str = 'normal'
    kd_mode: str = 'standard'
    description: str = ""


class AblationStudy:
    """Base class for ablation studies."""

    def __init__(self, study_name: str, description: str):
        self.study_name = study_name
        self.description = description
        self.models: List[ModelConfig] = []
        self.results: Dict[str, Dict[str, Any]] = {}

    def add_model(self, config: ModelConfig) -> None:
        """Add model to study."""
        self.models.append(config)
        logger.info(f"Added model: {config.name}")

    def run(self, device: str = 'auto') -> None:
        """Run evaluation on all models in study."""
        logger.info("=" * 80)
        logger.info(f"ABLATION STUDY: {self.study_name}")
        logger.info(f"Description: {self.description}")
        logger.info("=" * 80)

        for model_config in self.models:
            logger.info(f"\nEvaluating model: {model_config.name}")

            # Create args object for evaluation
            args = self._create_args(model_config, device)

            # Run evaluation
            try:
                config = EvaluationConfig(args)
                evaluator = Evaluator(config)
                results = evaluator.evaluate()
                self.results[model_config.name] = results
                logger.info(f"✓ Completed: {model_config.name}")
            except Exception as e:
                logger.error(f"✗ Failed: {model_config.name} - {e}")

    def _create_args(self, model_config: ModelConfig, device: str) -> argparse.Namespace:
        """Create args namespace for evaluation."""
        return argparse.Namespace(
            dataset=model_config.dataset,
            model_path=model_config.model_path,
            teacher_path=model_config.teacher_path,
            training_mode=model_config.training_mode,
            mode=model_config.kd_mode,
            batch_size=512,
            device=device,
            csv_output=None,
            json_output=None,
            plots_dir=None,
            threshold_optimization=True,
            verbose=False
        )

    def compute_deltas(self) -> pd.DataFrame:
        """
        Compute metric deltas between models.

        Returns DataFrame with columns: [model_a, model_b, metric, value_a, value_b, delta, pct_change, winner]
        """
        raise NotImplementedError("Subclasses must implement compute_deltas()")

    def export_results(self, output_dir: str = '.') -> None:
        """Export ablation results to CSV and JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export individual model results
        for model_name, results in self.results.items():
            json_path = output_path / f"{self.study_name}_{model_name}_results.json"
            with open(json_path, 'w') as f:
                json.dump(self._serialize_results(results), f, indent=2)
            logger.info(f"Exported: {json_path}")

        # Export deltas
        deltas_df = self.compute_deltas()
        csv_path = output_path / f"{self.study_name}_ablation.csv"
        deltas_df.to_csv(csv_path, index=False)
        logger.info(f"Exported: {csv_path}")

    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        import numpy as np

        serialized = {}
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_results(value)
            else:
                serialized[key] = value
        return serialized

    def print_summary(self) -> None:
        """Print ablation study summary."""
        logger.info("\n" + "=" * 80)
        logger.info(f"ABLATION STUDY SUMMARY: {self.study_name}")
        logger.info("=" * 80)
        deltas = self.compute_deltas()
        logger.info("\nModel Comparisons:")
        logger.info(deltas.to_string(index=False))


class KDImpactStudy(AblationStudy):
    """Knowledge Distillation Impact: With-KD vs Without-KD."""

    def __init__(self):
        super().__init__(
            study_name="kd_impact",
            description="Isolate impact of knowledge distillation on model performance"
        )

    def compute_deltas(self) -> pd.DataFrame:
        """Compute KD impact metrics."""
        rows = []

        # Group results by base model (assuming naming convention)
        no_kd_models = {k: v for k, v in self.results.items() if 'no_kd' in k.lower()}
        with_kd_models = {k: v for k, v in self.results.items() if 'with_kd' in k.lower()}

        for model_base in set([k.replace('_no_kd', '').replace('_with_kd', '') for k in self.results.keys()]):
            no_kd_key = f"{model_base}_no_kd"
            with_kd_key = f"{model_base}_with_kd"

            if no_kd_key not in self.results or with_kd_key not in self.results:
                continue

            # Extract test metrics for comparison
            no_kd_test = self.results[no_kd_key].get('test', {})
            with_kd_test = self.results[with_kd_key].get('test', {})

            # Get F1 scores
            no_kd_f1 = no_kd_test.get('classification', {}).get('f1', 0)
            with_kd_f1 = with_kd_test.get('classification', {}).get('f1', 0)

            delta_f1 = with_kd_f1 - no_kd_f1
            pct_change_f1 = (delta_f1 / no_kd_f1 * 100) if no_kd_f1 > 0 else 0

            rows.append({
                'study': 'kd_impact',
                'model_a': no_kd_key,
                'model_b': with_kd_key,
                'metric': 'f1',
                'value_a': no_kd_f1,
                'value_b': with_kd_f1,
                'delta': delta_f1,
                'pct_change': pct_change_f1,
                'winner': with_kd_key if delta_f1 > 0 else no_kd_key
            })

            # Get AUC scores
            no_kd_auc = no_kd_test.get('threshold_independent', {}).get('roc_auc', 0)
            with_kd_auc = with_kd_test.get('threshold_independent', {}).get('roc_auc', 0)

            delta_auc = with_kd_auc - no_kd_auc
            pct_change_auc = (delta_auc / no_kd_auc * 100) if no_kd_auc > 0 else 0

            rows.append({
                'study': 'kd_impact',
                'model_a': no_kd_key,
                'model_b': with_kd_key,
                'metric': 'auc_roc',
                'value_a': no_kd_auc,
                'value_b': with_kd_auc,
                'delta': delta_auc,
                'pct_change': pct_change_auc,
                'winner': with_kd_key if delta_auc > 0 else no_kd_key
            })

        return pd.DataFrame(rows)


class CurriculumImpactStudy(AblationStudy):
    """Curriculum Learning Impact: Curriculum vs Standard."""

    def __init__(self):
        super().__init__(
            study_name="curriculum_impact",
            description="Isolate impact of curriculum learning on model performance"
        )

    def compute_deltas(self) -> pd.DataFrame:
        """Compute curriculum impact metrics."""
        rows = []

        standard_models = {k: v for k, v in self.results.items() if 'standard' in k.lower()}
        curriculum_models = {k: v for k, v in self.results.items() if 'curriculum' in k.lower()}

        for model_base in set([k.replace('_standard', '').replace('_curriculum', '') for k in self.results.keys()]):
            standard_key = f"{model_base}_standard"
            curriculum_key = f"{model_base}_curriculum"

            if standard_key not in self.results or curriculum_key not in self.results:
                continue

            standard_test = self.results[standard_key].get('test', {})
            curriculum_test = self.results[curriculum_key].get('test', {})

            # Compare key metrics
            for metric_name in ['f1', 'recall', 'specificity', 'balanced_accuracy']:
                standard_val = standard_test.get('classification', {}).get(metric_name, 0)
                curriculum_val = curriculum_test.get('classification', {}).get(metric_name, 0)

                delta = curriculum_val - standard_val
                pct_change = (delta / standard_val * 100) if standard_val > 0 else 0

                rows.append({
                    'study': 'curriculum_impact',
                    'model_a': standard_key,
                    'model_b': curriculum_key,
                    'metric': metric_name,
                    'value_a': standard_val,
                    'value_b': curriculum_val,
                    'delta': delta,
                    'pct_change': pct_change,
                    'winner': curriculum_key if delta > 0 else standard_key
                })

        return pd.DataFrame(rows)


class FusionImpactStudy(AblationStudy):
    """Fusion Strategy Impact: DQN vs Static."""

    def __init__(self):
        super().__init__(
            study_name="fusion_impact",
            description="Isolate impact of fusion strategy (learned DQN vs static blending)"
        )

    def compute_deltas(self) -> pd.DataFrame:
        """Compute fusion impact metrics."""
        rows = []

        static_models = {k: v for k, v in self.results.items() if 'static' in k.lower()}
        dqn_models = {k: v for k, v in self.results.items() if 'dqn' in k.lower()}

        for model_base in set([k.replace('_static', '').replace('_dqn', '') for k in self.results.keys()]):
            static_key = f"{model_base}_static"
            dqn_key = f"{model_base}_dqn"

            if static_key not in self.results or dqn_key not in self.results:
                continue

            static_test = self.results[static_key].get('test', {})
            dqn_test = self.results[dqn_key].get('test', {})

            static_f1 = static_test.get('classification', {}).get('f1', 0)
            dqn_f1 = dqn_test.get('classification', {}).get('f1', 0)

            delta = dqn_f1 - static_f1
            pct_change = (delta / static_f1 * 100) if static_f1 > 0 else 0

            rows.append({
                'study': 'fusion_impact',
                'model_a': static_key,
                'model_b': dqn_key,
                'metric': 'f1',
                'value_a': static_f1,
                'value_b': dqn_f1,
                'delta': delta,
                'pct_change': pct_change,
                'winner': dqn_key if delta > 0 else static_key
            })

        return pd.DataFrame(rows)


class TrainingModeImpactStudy(AblationStudy):
    """Training Mode Impact: Compare different training modes."""

    def __init__(self):
        super().__init__(
            study_name="training_mode_impact",
            description="Compare different training modes: normal, autoencoder, curriculum"
        )

    def compute_deltas(self) -> pd.DataFrame:
        """Compute training mode impact metrics."""
        rows = []
        modes = ['normal', 'autoencoder', 'curriculum']

        # Create comparison matrix
        for i, mode_a in enumerate(modes):
            for mode_b in modes[i + 1:]:
                mode_a_models = {k: v for k, v in self.results.items() if mode_a in k.lower()}
                mode_b_models = {k: v for k, v in self.results.items() if mode_b in k.lower()}

                if not mode_a_models or not mode_b_models:
                    continue

                # Pick first model of each mode for comparison
                model_a_key = list(mode_a_models.keys())[0]
                model_b_key = list(mode_b_models.keys())[0]

                test_a = self.results[model_a_key].get('test', {})
                test_b = self.results[model_b_key].get('test', {})

                f1_a = test_a.get('classification', {}).get('f1', 0)
                f1_b = test_b.get('classification', {}).get('f1', 0)

                delta = f1_b - f1_a
                pct_change = (delta / f1_a * 100) if f1_a > 0 else 0

                rows.append({
                    'study': 'training_mode_impact',
                    'model_a': model_a_key,
                    'model_b': model_b_key,
                    'metric': 'f1',
                    'value_a': f1_a,
                    'value_b': f1_b,
                    'delta': delta,
                    'pct_change': pct_change,
                    'winner': model_b_key if delta > 0 else model_a_key
                })

        return pd.DataFrame(rows)


def main():
    """Main entry point for ablation studies."""
    parser = argparse.ArgumentParser(
        description='Run ablation studies to compare CAN-Graph model variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run KD impact study
  python -m src.evaluation.ablation \\
    --study kd_impact \\
    --model-list models_kd.json \\
    --output-dir ablation_results/

  # Run curriculum impact study
  python -m src.evaluation.ablation \\
    --study curriculum \\
    --model-list models_curriculum.json \\
    --output-dir ablation_results/
        '''
    )

    parser.add_argument('--study', required=True,
                       choices=['kd', 'curriculum', 'fusion', 'training_mode'],
                       help='Which ablation study to run')
    parser.add_argument('--model-list', required=True,
                       help='JSON file with model configurations')
    parser.add_argument('--output-dir', default='ablation_results',
                       help='Directory to save results')
    parser.add_argument('--device', default='auto',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use for evaluation')

    args = parser.parse_args()

    # Load model configurations
    try:
        with open(args.model_list, 'r') as f:
            model_configs = json.load(f)
        logger.info(f"Loaded {len(model_configs)} model configurations")
    except Exception as e:
        logger.error(f"Failed to load model list: {e}")
        return 1

    # Select and run study
    if args.study == 'kd':
        study = KDImpactStudy()
    elif args.study == 'curriculum':
        study = CurriculumImpactStudy()
    elif args.study == 'fusion':
        study = FusionImpactStudy()
    elif args.study == 'training_mode':
        study = TrainingModeImpactStudy()
    else:
        logger.error(f"Unknown study: {args.study}")
        return 1

    # Add models to study
    for config_dict in model_configs:
        config = ModelConfig(**config_dict)
        study.add_model(config)

    # Run ablation
    try:
        study.run(device=args.device)
        study.print_summary()
        study.export_results(output_dir=args.output_dir)
        logger.info("Ablation study completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
