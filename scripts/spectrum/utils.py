"""Utility functions for political spectrum analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np


def setup_plot_style():
    """Setup matplotlib/seaborn plotting style."""
    plt.style.use('seaborn')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300


def save_plot(plt, filepath: Path, show_plots: bool = False):
    """Save and optionally display a plot.
    
    Args:
        plt: matplotlib.pyplot instance
        filepath: Path object for save location
        show_plots: bool, whether to display plot
    """
    plt.tight_layout()
    plt.savefig(filepath)
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def scale_features(data):
    """Scale features to standard normal distribution."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def create_3d_plot():
    """Create a 3D plot figure."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax 