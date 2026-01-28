"""
Data loading utilities for visualization scripts.

Uses config-driven approach to load datasets consistently with training/evaluation.

This ensures visualizations use the exact same data as the models were trained/evaluated on.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config.frozen_config import load_frozen_config
from src.data.dataset_handler import CANDatasetHandler
from src.config.hydra_zen_configs import CANGraphConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Utility class for loading datasets for visualization.

    Uses frozen configs to ensure consistency with training/evaluation.

    Example:
        data_loader = DataLoader()
        train_data, val_data, test_data = data_loader.load_dataset_from_config(
            config_path="experimentruns/.../configs/frozen_config.json"
        )
    """

    def __init__(self):
        """Initialize data loader."""
        pass

    def load_dataset_from_config(
        self,
        config_path: str,
        splits: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Load dataset using frozen config.

        Args:
            config_path: Path to frozen config JSON
            splits: List of splits to load ['train', 'val', 'test'] (default: all)
            max_samples_per_split: Optional limit on samples per split (for quick testing)

        Returns:
            Dictionary mapping split_name -> list of PyG Data objects

        Example:
            data = data_loader.load_dataset_from_config(
                config_path="experimentruns/.../configs/frozen_config.json",
                splits=['val', 'test']
            )
            val_data = data['val']
            test_data = data['test']
        """
        logger.info(f"Loading dataset from config: {config_path}")

        # Load frozen config
        config = load_frozen_config(config_path)

        # Initialize dataset handler
        dataset_handler = CANDatasetHandler(config)

        # Load datasets
        if splits is None:
            splits = ['train', 'val', 'test']

        datasets = {}

        for split in splits:
            logger.info(f"Loading {split} split...")

            if split == 'train':
                data_list = dataset_handler.train_dataset
            elif split == 'val':
                data_list = dataset_handler.val_dataset
            elif split == 'test':
                data_list = dataset_handler.test_dataset
            else:
                raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

            # Limit samples if requested
            if max_samples_per_split is not None and len(data_list) > max_samples_per_split:
                logger.info(f"Limiting {split} to {max_samples_per_split} samples")
                # Random sample for diversity
                indices = np.random.choice(len(data_list), max_samples_per_split, replace=False)
                data_list = [data_list[i] for i in indices]

            datasets[split] = data_list
            logger.info(f"Loaded {len(data_list)} samples for {split}")

        return datasets

    def load_dataset_by_name(
        self,
        dataset_name: str,
        modality: str = 'automotive',
        data_path: Optional[str] = None,
        splits: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Load dataset by name (without config).

        Args:
            dataset_name: Name of dataset ('hcrl_sa', 'hcrl_ch', etc.)
            modality: 'automotive' or other
            data_path: Optional explicit data path (default: data/{modality}/{dataset_name})
            splits: List of splits to load
            max_samples_per_split: Optional limit on samples per split

        Returns:
            Dictionary mapping split_name -> list of PyG Data objects

        Example:
            data = data_loader.load_dataset_by_name(
                dataset_name='hcrl_sa',
                splits=['val', 'test'],
                max_samples_per_split=5000
            )
        """
        logger.info(f"Loading dataset: {dataset_name}")

        # Create minimal config for dataset loading
        from src.config.hydra_zen_configs import (
            CANDatasetConfig,
            VGAEConfig,
            AutoencoderTrainingConfig
        )

        if data_path is None:
            data_path = f"data/{modality}/{dataset_name}"

        # Create config
        dataset_config = CANDatasetConfig(
            name=dataset_name,
            modality=modality,
            data_path=data_path
        )

        # Minimal model and training configs (required by CANGraphConfig)
        model_config = VGAEConfig()  # Default VGAE config
        training_config = AutoencoderTrainingConfig()  # Default training config

        # Create full config
        config = CANGraphConfig(
            dataset=dataset_config,
            model=model_config,
            training=training_config
        )

        # Initialize dataset handler
        dataset_handler = CANDatasetHandler(config)

        # Load datasets
        if splits is None:
            splits = ['train', 'val', 'test']

        datasets = {}

        for split in splits:
            logger.info(f"Loading {split} split...")

            if split == 'train':
                data_list = dataset_handler.train_dataset
            elif split == 'val':
                data_list = dataset_handler.val_dataset
            elif split == 'test':
                data_list = dataset_handler.test_dataset
            else:
                raise ValueError(f"Unknown split: {split}")

            # Limit samples if requested
            if max_samples_per_split is not None and len(data_list) > max_samples_per_split:
                logger.info(f"Limiting {split} to {max_samples_per_split} samples")
                indices = np.random.choice(len(data_list), max_samples_per_split, replace=False)
                data_list = [data_list[i] for i in indices]

            datasets[split] = data_list
            logger.info(f"Loaded {len(data_list)} samples for {split}")

        return datasets

    def get_attack_type_labels(
        self,
        data_list: List,
        dataset_name: str = 'hcrl_sa'
    ) -> np.ndarray:
        """
        Extract attack type labels from data.

        Args:
            data_list: List of PyG Data objects
            dataset_name: Dataset name for attack-specific mapping

        Returns:
            Array of attack type labels (0=normal, 1+=attack types)

        Note:
            Currently simplified to binary (0=normal, 1=attack).
            TODO: Add per-attack-type labeling from metadata.
        """
        labels = np.array([data.y.item() for data in data_list])
        return labels

    def get_class_distribution(
        self,
        data_list: List
    ) -> Dict[int, int]:
        """
        Get class distribution from dataset.

        Args:
            data_list: List of PyG Data objects

        Returns:
            Dictionary mapping class_label -> count
        """
        labels = [data.y.item() for data in data_list]
        unique, counts = np.unique(labels, return_counts=True)

        distribution = {int(label): int(count) for label, count in zip(unique, counts)}

        logger.info(f"Class distribution: {distribution}")

        return distribution


def load_evaluation_results(
    results_dir: str,
    model_names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load evaluation results from CSV files.

    Args:
        results_dir: Directory containing evaluation CSV files
        model_names: Optional list of specific model names to load (default: all)

    Returns:
        Dictionary mapping model_name -> DataFrame with metrics

    Example:
        results = load_evaluation_results('evaluation_results/hcrl_sa/teacher')
        vgae_metrics = results['vgae_teacher_autoencoder_run_003']
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.warning(f"Results directory '{results_dir}' does not exist")
        return results

    # Find all CSV files
    csv_files = list(results_path.glob('*.csv'))

    for csv_file in csv_files:
        model_name = csv_file.stem

        # Filter if model_names specified
        if model_names is not None and model_name not in model_names:
            continue

        try:
            df = pd.read_csv(csv_file)
            results[model_name] = df
            logger.info(f"Loaded results for {model_name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")

    return results


def load_dqn_predictions(
    predictions_file: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DQN predictions with 15D states and alpha values.

    Args:
        predictions_file: Path to saved predictions (.npz or .pkl)

    Returns:
        Tuple of (states_15d, alpha_values, predictions, labels):
        - states_15d: [N, 15] array of state features
        - alpha_values: [N] array of selected fusion weights
        - predictions: [N] array of final predictions (0 or 1)
        - labels: [N] array of ground truth labels

    Example:
        states, alphas, preds, labels = load_dqn_predictions(
            'evaluation_results/hcrl_sa/fusion/dqn_predictions.npz'
        )
    """
    if predictions_file.endswith('.npz'):
        data = np.load(predictions_file)
        return (
            data['states'],
            data['alphas'],
            data['predictions'],
            data['labels']
        )
    elif predictions_file.endswith('.pkl') or predictions_file.endswith('.pickle'):
        import pickle
        with open(predictions_file, 'rb') as f:
            data = pickle.load(f)
        return (
            data['states'],
            data['alphas'],
            data['predictions'],
            data['labels']
        )
    else:
        raise ValueError(f"Unsupported file format: {predictions_file}")


# Convenience function
def load_data_for_visualization(
    config_path: str,
    splits: Optional[List[str]] = None,
    max_samples: Optional[int] = 5000
) -> Dict[str, List]:
    """
    Convenience function to load data for visualization.

    Args:
        config_path: Path to frozen config
        splits: Which splits to load (default: ['val', 'test'])
        max_samples: Max samples per split (default: 5000 for efficiency)

    Returns:
        Dictionary mapping split_name -> data_list

    Example:
        data = load_data_for_visualization(
            config_path="experimentruns/.../configs/frozen_config.json",
            splits=['test'],
            max_samples=10000
        )
        test_data = data['test']
    """
    if splits is None:
        splits = ['val', 'test']

    loader = DataLoader()
    return loader.load_dataset_from_config(
        config_path=config_path,
        splits=splits,
        max_samples_per_split=max_samples
    )
