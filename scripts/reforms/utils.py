"""Utility functions for reform analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plot_style():
    """Setup matplotlib/seaborn plotting style."""
    plt.style.use('seaborn')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300


def save_plot(plt, filepath: Path, show: bool = False):
    """Save and optionally display a plot."""
    plt.tight_layout()
    plt.savefig(filepath)
    
    if show:
        plt.show()
    else:
        plt.close() 