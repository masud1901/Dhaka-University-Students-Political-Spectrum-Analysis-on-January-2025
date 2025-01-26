"""Configuration settings for DU student survey analysis."""
import os
from pathlib import Path

# Get the absolute path of the project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Base directories
SCRIPTS_DIR = ROOT_DIR / 'scripts'
DATA_DIR = ROOT_DIR / 'data'

# Data directories
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Raw data file paths
RAW_DATA_FILE = RAW_DATA_DIR / 'DU_Akib.csv'
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / 'cleaned_student_survey.csv'

# Analysis module directories
DEMOGRAPHICS_DIR = SCRIPTS_DIR / 'demographics'
PERSPECTIVES_DIR = SCRIPTS_DIR / 'perspectives'
REFORMS_DIR = SCRIPTS_DIR / 'reforms'
STATISTICS_DIR = SCRIPTS_DIR / 'statistics'

# Logs directory
LOGS_DIR = ROOT_DIR / 'logs'

# Visualization settings
FIGURE_SIZE = (12, 8)
STYLE_SHEET = 'seaborn'
COLOR_PALETTE = 'husl'

# Analysis settings
RANDOM_SEED = 42


def verify_data_file():
    """Verify that the raw data file exists."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nData file not found at {RAW_DATA_FILE}\n"
            f"Please place your data file at: {RAW_DATA_FILE}"
        )


def setup_directories():
    """Create all necessary directories."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True) 