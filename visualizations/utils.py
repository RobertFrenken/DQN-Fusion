"""
Common utilities for visualization generation.

Provides data loading, color palettes, and figure saving functions
to ensure consistency across all publication figures.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt

# Optional: seaborn for additional palettes
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Some palette features unavailable.")


# ============================================================================
# Color Palettes
# ============================================================================

# Colorblind-friendly palette (from Seaborn)
COLORBLIND_PALETTE = [
    '#0173B2',  # Blue
    '#DE8F05',  # Orange
    '#029E73',  # Green
    '#CC78BC',  # Purple
    '#CA9161',  # Brown
    '#ECE133',  # Yellow
    '#56B4E9',  # Light blue
    '#949494',  # Gray
]

# Model-specific colors
MODEL_COLORS = {
    'vgae': '#0173B2',        # Blue
    'gat': '#DE8F05',         # Orange
    'fusion': '#CC78BC',      # Purple
    'fusion_15d': '#9467BD',  # Dark purple (for 15D enhancement)
    'teacher': '#2CA02C',     # Green
    'student': '#FF7F0E',     # Orange
    'student_kd': '#1F77B4',  # Blue
}

# Class colors
CLASS_COLORS = {
    'normal': '#029E73',      # Green
    'attack': '#D62728',      # Red
}

# Confidence/quality gradient
CONFIDENCE_CMAP = 'RdYlGn'  # Red (low) to Yellow to Green (high)


def get_color_palette(palette_name: str = 'colorblind') -> List[str]:
    """
    Get a color palette by name.

    Args:
        palette_name: 'colorblind', 'model', 'class', or seaborn palette name

    Returns:
        List of color hex codes
    """
    if palette_name == 'colorblind':
        return COLORBLIND_PALETTE
    elif palette_name == 'model':
        return list(MODEL_COLORS.values())
    elif palette_name == 'class':
        return list(CLASS_COLORS.values())
    else:
        # Try seaborn palette
        if HAS_SEABORN:
            try:
                return sns.color_palette(palette_name, as_cmap=False).as_hex()
            except:
                print(f"Warning: Unknown palette '{palette_name}', using colorblind")
                return COLORBLIND_PALETTE
        else:
            print(f"Warning: Seaborn not available, using colorblind palette")
            return COLORBLIND_PALETTE


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_evaluation_results(results_dir: str = 'test_results') -> Dict[str, pd.DataFrame]:
    """
    Load all evaluation results from CSV files.

    Args:
        results_dir: Directory containing evaluation CSV files

    Returns:
        Dictionary mapping model_name -> DataFrame with metrics

    Example:
        results = load_evaluation_results()
        vgae_df = results['vgae_teacher_set01']
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: Results directory '{results_dir}' does not exist")
        return results

    # Find all CSV files
    csv_files = list(results_path.glob('*.csv'))

    for csv_file in csv_files:
        model_name = csv_file.stem  # Filename without extension
        try:
            df = pd.read_csv(csv_file)
            results[model_name] = df
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")

    return results


def load_json_results(results_dir: str = 'test_results') -> Dict[str, Dict]:
    """
    Load all evaluation results from JSON files.

    Args:
        results_dir: Directory containing evaluation JSON files

    Returns:
        Dictionary mapping model_name -> nested dict with full metrics
    """
    results = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Warning: Results directory '{results_dir}' does not exist")
        return results

    # Find all JSON files
    json_files = list(results_path.glob('*.json'))

    for json_file in json_files:
        model_name = json_file.stem
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            results[model_name] = data
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def load_dqn_predictions(predictions_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DQN predictions with 15D states and alpha values.

    Args:
        predictions_file: Path to saved predictions (numpy .npz or pickle)

    Returns:
        Tuple of (states_15d, alpha_values, predictions)
        - states_15d: [N, 15] array of state features
        - alpha_values: [N] array of selected fusion weights
        - predictions: [N] array of final predictions (0 or 1)
    """
    if predictions_file.endswith('.npz'):
        data = np.load(predictions_file)
        return data['states'], data['alphas'], data['predictions']
    elif predictions_file.endswith('.pkl') or predictions_file.endswith('.pickle'):
        import pickle
        with open(predictions_file, 'rb') as f:
            data = pickle.load(f)
        return data['states'], data['alphas'], data['predictions']
    else:
        raise ValueError(f"Unsupported file format: {predictions_file}")


def load_embeddings(embeddings_file: str, model_type: str = 'vgae') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load saved embeddings from VGAE or GAT models.

    Args:
        embeddings_file: Path to embeddings file (.npz)
        model_type: 'vgae' or 'gat'

    Returns:
        Tuple of (embeddings, labels)
        - embeddings: [N, D] array of embedding vectors
        - labels: [N] array of class labels (0=normal, 1=attack)
    """
    data = np.load(embeddings_file)

    if model_type == 'vgae':
        # VGAE latent space (z)
        embeddings = data['latent_z']  # [N, latent_dim] e.g., [N, 48]
    elif model_type == 'gat':
        # GAT pre-pooling embeddings
        embeddings = data['pre_pooling_embeddings']  # [N, hidden_dim]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    labels = data['labels']

    return embeddings, labels


def load_training_logs(log_dir: str, model_name: str) -> pd.DataFrame:
    """
    Load training logs from MLflow or CSV.

    Args:
        log_dir: Directory containing training logs
        model_name: Name of model to load logs for

    Returns:
        DataFrame with columns: [epoch, train_loss, val_loss, ...]
    """
    # Try MLflow format first
    mlflow_path = Path(log_dir) / 'mlruns'
    if mlflow_path.exists():
        # TODO: Implement MLflow parsing
        pass

    # Try CSV format
    csv_path = Path(log_dir) / f"{model_name}_training_log.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    print(f"Warning: No training logs found for {model_name}")
    return pd.DataFrame()


# ============================================================================
# Figure Saving and Management
# ============================================================================

def save_figure(fig: plt.Figure,
                filename: str,
                output_dir: str = 'figures',
                formats: List[str] = ['pdf', 'png'],
                dpi: int = 300,
                tight: bool = True) -> List[str]:
    """
    Save figure in multiple formats with consistent settings.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save ('pdf', 'png', 'svg', 'eps')
        dpi: Resolution for raster formats
        tight: Use tight_layout

    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if tight:
        fig.tight_layout()

    saved_files = []

    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"

        # Format-specific settings
        save_kwargs = {'dpi': dpi, 'bbox_inches': 'tight'}

        if fmt == 'pdf':
            save_kwargs['backend'] = 'pdf'
        elif fmt == 'svg':
            save_kwargs['format'] = 'svg'
        elif fmt == 'eps':
            save_kwargs['format'] = 'eps'

        fig.savefig(filepath, **save_kwargs)
        saved_files.append(str(filepath))
        print(f"✓ Saved: {filepath}")

    return saved_files


def setup_figure(width: float = 7.0,
                height: float = 4.5,
                nrows: int = 1,
                ncols: int = 1,
                style: str = '../paper_style.mplstyle',
                **subplot_kw) -> Tuple[plt.Figure, Any]:
    """
    Create figure with publication settings.

    Args:
        width: Figure width in inches
        height: Figure height in inches
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        style: Path to style file or matplotlib style name
        **subplot_kw: Additional kwargs for subplots

    Returns:
        Tuple of (figure, axes)
    """
    # Load style
    if os.path.exists(style):
        plt.style.use(style)
    else:
        print(f"Warning: Style file '{style}' not found, using default")

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **subplot_kw)

    return fig, axes


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_confidence_intervals(data: np.ndarray,
                                confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval.

    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    import scipy.stats as stats

    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)

    return mean, ci[0], ci[1]


def annotate_bars(ax: plt.Axes,
                 bar_container,
                 values: Optional[np.ndarray] = None,
                 fmt: str = '.2f',
                 offset: float = 0.01) -> None:
    """
    Add value labels on top of bars.

    Args:
        ax: Matplotlib axes
        bar_container: Container of bar patches
        values: Optional array of values (if None, uses bar heights)
        fmt: Format string for labels
        offset: Vertical offset as fraction of y-range
    """
    for i, bar in enumerate(bar_container):
        height = bar.get_height()
        value = values[i] if values is not None else height

        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{value:{fmt}}',
                ha='center', va='bottom', fontsize=9)


# ============================================================================
# Dataset and Attack Type Mapping
# ============================================================================

DATASET_NAMES = {
    'hcrl_sa': 'HCRL-SA',
    'hcrl_ch': 'HCRL-CH',
    'set_01': 'Set 01',
    'set_02': 'Set 02',
    'set_03': 'Set 03',
    'set_04': 'Set 04',
}

ATTACK_TYPES = {
    'dos': 'Denial of Service',
    'fuzzing': 'Fuzzing',
    'rpm': 'RPM Spoofing',
    'gear': 'Gear Spoofing',
    'normal': 'Normal Traffic',
}


def get_attack_type_from_label(label: int, dataset: str = 'hcrl_sa') -> str:
    """
    Map numeric label to attack type name.

    Args:
        label: 0 (normal) or 1 (attack)
        dataset: Dataset name for attack-specific mapping

    Returns:
        Human-readable attack type name
    """
    if label == 0:
        return 'Normal'
    else:
        return 'Attack'  # Generic, refine based on dataset if needed


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    print("CAN-Graph Visualization Utilities")
    print("=" * 50)

    # Test color palette
    colors = get_color_palette('colorblind')
    print(f"\nColorblind palette ({len(colors)} colors):")
    for i, color in enumerate(colors):
        print(f"  {i}: {color}")

    # Test figure setup
    fig, ax = setup_figure(width=6, height=4)
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Test Figure')

    # Save figure
    saved = save_figure(fig, 'test_figure', output_dir='../figures/test')
    print(f"\nSaved test figure:")
    for path in saved:
        print(f"  {path}")

    plt.close(fig)
    print("\n✓ Utilities test complete!")
