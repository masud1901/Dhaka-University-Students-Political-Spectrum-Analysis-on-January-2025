"""Utility functions for perspective analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plot_style():
    """Setup matplotlib/seaborn plotting style."""
    plt.style.use('seaborn')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300


def save_plot(plt, filepath: Path, show_plots: bool = False):
    """Save and optionally display a plot.
    
    Args:
        plt: matplotlib pyplot instance
        filepath: Path object for save location
        show_plots: bool, whether to display plot
    """
    plt.tight_layout()
    plt.savefig(filepath)
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def calculate_percentage(data, column: str, condition) -> float:
    """Calculate percentage of data meeting a condition.
    
    Args:
        data: pandas DataFrame
        column: str, column name to check
        condition: function to apply to column
        
    Returns:
        float: percentage of rows meeting condition
    """
    if column not in data.columns:
        return 0.0
    return (data[column].apply(condition)).mean() * 100 