"""
Publication-Ready Plotting Configuration
Contains matplotlib settings and color schemes for scientific paper style plots.
"""
import matplotlib.pyplot as plt

# Publication-Ready Plotting Configuration (Scientific Paper Style)
PLOT_CONFIG = {
    # Font settings - clean, professional
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    
    # Figure settings - high DPI, clean background
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'none',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    
    # Line and marker settings - thin, precise
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'patch.linewidth': 0.8,
    
    # Grid and axes - minimal, clean
    'axes.grid': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.axisbelow': True,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.facecolor': 'white',
    
    # Ticks - clean, minimal
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    
    # Colors - scientific, muted palette
    'axes.prop_cycle': plt.cycler('color', [
        '#2E86AB',  # Blue
        '#A23B72',  # Magenta  
        '#F18F01',  # Orange
        '#C73E1D',  # Red
        '#592E83',  # Purple
        '#1B5F40',  # Green
        '#8B4513',  # Brown
        '#708090'   # Slate Gray
    ]),
    
    # LaTeX rendering
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',
    
    # Legend settings - clean, minimal
    'legend.frameon': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1
}

# Color schemes for different plot types (Scientific Paper Style)
COLOR_SCHEMES = {
    'training': {
        'accuracy': '#2E86AB',
        'reward': '#1B5F40', 
        'loss': '#C73E1D',
        'q_values': '#592E83'
    },
    'validation': {
        'primary': '#A23B72',
        'secondary': '#8B4513'
    },
    'fusion_analysis': {
        'normal': '#2E86AB',
        'attack': '#A23B72',
        'adaptive': '#F18F01'
    },
    'heatmap': 'RdYlBu_r',
    'contour': 'viridis'
}

def apply_publication_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update(PLOT_CONFIG)
    plt.style.use('default')  # Reset to default first
    for key, value in PLOT_CONFIG.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value

def save_publication_figure(fig, filename, additional_formats=None):
    """Save figure in multiple publication-ready formats.
    
    Args:
        fig: matplotlib figure object
        filename: base filename with extension
        additional_formats: list of additional formats to save ['pdf', 'svg', 'eps'] or None for PNG only
    """
    base_path = filename.rsplit('.', 1)[0]
    
    # Always save PNG (default)
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    
    # Save additional formats only if specified
    if additional_formats:
        for fmt in additional_formats:
            try:
                fig.savefig(f"{base_path}.{fmt}", bbox_inches='tight', pad_inches=0.05)
            except Exception as e:
                print(f"Warning: Could not save {fmt} format: {e}")