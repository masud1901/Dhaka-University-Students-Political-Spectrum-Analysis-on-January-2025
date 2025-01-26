"""Utility functions for DU student survey analysis."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from . import config


def load_data(processed: bool = True) -> pd.DataFrame:
    """Load survey data from CSV file.
    
    Args:
        processed: bool, whether to load processed or raw data
    
    Returns:
        pd.DataFrame: loaded survey data
    """
    # Verify data file exists
    config.verify_data_file()
    
    # Load appropriate data file
    file_path = (
        config.CLEANED_DATA_FILE if processed
        else config.RAW_DATA_FILE
    )
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from: {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def setup_plotting_style():
    """Setup matplotlib/seaborn plotting style."""
    plt.style.use(config.STYLE_SHEET)
    sns.set_palette(config.COLOR_PALETTE)
    plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
    plt.rcParams['figure.dpi'] = 300


def save_plot(plt, filepath: Path, show: bool = False):
    """Save and optionally display a plot."""
    plt.tight_layout()
    plt.savefig(filepath)
    
    if show:
        plt.show()
    else:
        plt.close()


def save_analysis_results(results: dict, analysis_type: str):
    """Save analysis results to file.
    
    Args:
        results: dict containing analysis results
        analysis_type: str name of analysis type
    """
    output_dir = config.LOGS_DIR / analysis_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_file = output_dir / f'{analysis_type}_results.txt'
    with open(output_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}:\n")
            f.write(f"{str(value)}\n\n")