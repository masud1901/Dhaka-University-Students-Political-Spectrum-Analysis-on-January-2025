"""Main analysis pipeline for DU student survey data."""
from datetime import datetime
import sys
from io import StringIO
from . import config, utils
from .cleaners import StudentSurveyDataCleaner
from .demographics import DemographicAnalyzer
from .political import PoliticalAnalyzer
from .reforms import ReformAnalyzer
from .values import ValueAnalyzer
from .spectrum import SpectrumAnalyzer
from .perspectives import PerspectiveAnalyzer


class OutputCapture:
    """Capture stdout to both display and save to file."""
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        self.output = StringIO()

    def write(self, text):
        self.terminal.write(text)
        self.output.write(text)

    def flush(self):
        self.terminal.flush()

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(self.output.getvalue())


def setup_directories():
    """Create necessary directories if they don't exist."""
    config.setup_directories()


def run_analysis_pipeline(show_plots: bool = False):
    """Run complete analysis pipeline."""
    # Setup output capture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = config.LOGS_DIR / f'analysis_report_{timestamp}.txt'
    output_capture = OutputCapture(output_file)
    sys.stdout = output_capture

    try:
        # Setup
        setup_directories()
        utils.setup_plotting_style()

        print("Starting DU student survey analysis...")

        # Initialize data cleaner and clean data
        print("\nLoading and cleaning data...")
        data_cleaner = StudentSurveyDataCleaner()
        raw_data = utils.load_data(processed=False)
        cleaned_data = data_cleaner.run_data_cleaning(raw_data)

        # Run demographic analysis
        print("\nRunning demographic analysis...")
        demographic_analyzer = DemographicAnalyzer(cleaned_data)
        demographic_results = demographic_analyzer.run_demographic_analysis(
            show_plots=show_plots
        )

        # Run political analysis
        print("\nRunning political spectrum analysis...")
        political_analyzer = PoliticalAnalyzer(cleaned_data)
        political_results = political_analyzer.run_political_analysis(
            show_plots=show_plots
        )

        # Run reform analysis
        print("\nRunning reform analysis...")
        reform_analyzer = ReformAnalyzer(cleaned_data)
        reform_results = reform_analyzer.run_reform_analysis(
            show_plots=show_plots
        )

        # Run value analysis
        print("\nRunning value analysis...")
        value_analyzer = ValueAnalyzer(cleaned_data)
        value_results = value_analyzer.run_value_analysis(
            show_plots=show_plots
        )

        # Run spectrum analysis
        print("\nRunning political spectrum analysis...")
        spectrum_analyzer = SpectrumAnalyzer(cleaned_data)
        spectrum_results = spectrum_analyzer.run_spectrum_analysis(
            show_plots=show_plots
        )

        # Run perspective analysis
        print("\nRunning perspective analysis...")
        perspective_analyzer = PerspectiveAnalyzer(cleaned_data)
        perspective_results = perspective_analyzer.run_perspective_analysis(
            show_plots=show_plots
        )

        # Combine all results
        results = {
            'demographic': demographic_results,
            'political': political_results,
            'reform': reform_results,
            'value': value_results,
            'spectrum': spectrum_results,
            'perspective': perspective_results
        }

        print("\nAnalysis completed successfully!")
        print(f"Analysis report saved to: {output_file}")

        # Save captured output
        output_capture.save()

        return results
    
    finally:
        # Restore original stdout
        sys.stdout = output_capture.terminal


if __name__ == "__main__":
    run_analysis_pipeline()