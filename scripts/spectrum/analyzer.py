"""Complex political spectrum analysis module."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .utils import setup_plot_style, save_plot
from .cluster_analyzer import ClusterAnalyzer
from .dimension_analyzer import DimensionAnalyzer
from .stats_analyzer import StatsAnalyzer


class SpectrumAnalyzer:
    """Analyzes complex political spectrum data."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with cleaned survey data."""
        self.data = data
        self.output_dir = Path(__file__).parent
        self.figures_dir = self.output_dir / 'figures'
        self.reports_dir = self.output_dir / 'reports'
        
        # Create output directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting style
        setup_plot_style()
        
        # Define feature sets for left-right spectrum
        self.spectrum_indicators = {
            'economic': {
                'right_wing': [
                    'market_over_regulation',  # 1 indicates market preference
                    'economic_growth_over_environment',  # 1 indicates growth priority
                    'economic_freedom_score'  # Higher score indicates free market
                ],
                'weight': 0.4  # Economic factors weight
            },
            'social': {
                'right_wing': [
                    'individual_freedom_over_harmony',  # 1 indicates individualism
                    'stability_over_justice',  # 1 indicates stability preference
                    'central_over_local'  # 1 indicates centralization
                ],
                'weight': 0.3  # Social factors weight
            },
            'cultural': {
                'right_wing': [
                    'traditional_over_progressive',  # 1 indicates traditionalism
                    'tradition_reliance_score'  # Higher score indicates traditionalism
                ],
                'weight': 0.3  # Cultural factors weight
            }
        }

    def calculate_spectrum_score(self) -> pd.DataFrame:
        """Calculate political spectrum score (0=left, 1=right)."""
        scores = pd.DataFrame()
        
        # Calculate dimensional scores
        for dimension, config in self.spectrum_indicators.items():
            indicators = config['right_wing']
            weight = config['weight']
            
            # Get available indicators
            available_indicators = [
                col for col in indicators 
                if col in self.data.columns
            ]
            
            if not available_indicators:
                print(f"Warning: No indicators available for {dimension}")
                continue
            
            # Normalize each indicator to 0-1 scale
            normalized_scores = []
            for indicator in available_indicators:
                series = self.data[indicator]
                if series.dtype == bool:
                    # Binary indicators (already 0 or 1)
                    normalized = series.astype(float)
                else:
                    # Scale numeric indicators to 0-1
                    normalized = (series - series.min()) / (
                        series.max() - series.min()
                    )
                normalized_scores.append(normalized)
            
            # Calculate weighted average for dimension
            if normalized_scores:
                dimension_score = sum(normalized_scores) / len(normalized_scores)
                scores[f'{dimension}_spectrum'] = dimension_score * weight
        
        # Calculate overall spectrum score
        if not scores.empty:
            scores['spectrum_score'] = scores.sum(axis=1)
            
            # Ensure final score is between 0 and 1
            scores['spectrum_score'] = scores['spectrum_score'].clip(0, 1)
        
        return scores

    def analyze_spectrum_distribution(self, scores: pd.DataFrame, show_plots: bool = False):
        """Analyze the distribution of spectrum scores."""
        import seaborn as sns
        
        # Create distribution plot
        plt.figure(figsize=(12, 6))
        
        # Main distribution
        sns.histplot(
            data=scores['spectrum_score'],
            bins=30,
            kde=True
        )
        plt.title('Distribution of Political Spectrum Scores')
        plt.xlabel('Spectrum Score (0=Left, 1=Right)')
        plt.ylabel('Count')
        
        # Add mean and median lines
        mean_score = scores['spectrum_score'].mean()
        median_score = scores['spectrum_score'].median()
        plt.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='g', linestyle='--', label=f'Median: {median_score:.2f}')
        plt.legend()
        
        # Save plot
        plt.savefig(self.figures_dir / 'spectrum_distribution.png')
        if not show_plots:
            plt.close()
        
        # Calculate distribution statistics
        stats = {
            'mean': mean_score,
            'median': median_score,
            'std': scores['spectrum_score'].std(),
            'quartiles': scores['spectrum_score'].quantile([0.25, 0.75]).to_dict()
        }
        
        return stats

    def generate_summary_report(self, results: dict):
        """Generate summary report of spectrum analysis."""
        report = [
            "Political Spectrum Analysis Summary",
            "================================\n"
        ]
        
        # Dimension analysis summary
        if results.get('dimension_analysis'):
            report.extend([
                "Dimensionality Analysis:",
                "---------------------",
                results['dimension_analysis'].get('summary', 'No details available')
            ])
        
        # Cluster analysis summary
        if results.get('cluster_analysis'):
            report.extend([
                "\nClustering Analysis:",
                "------------------",
                results['cluster_analysis'].get('summary', 'No details available')
            ])
        
        # Save report
        report_path = self.reports_dir / 'spectrum_summary.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def run_spectrum_analysis(self, show_plots: bool = False) -> dict:
        """Run complete political spectrum analysis."""
        print("Running complex political spectrum analysis...")
        
        # Calculate spectrum scores
        scores = self.calculate_spectrum_score()
        if scores.empty:
            return {'error': "Could not calculate spectrum scores"}
        
        # Analyze spectrum distribution
        distribution_stats = self.analyze_spectrum_distribution(scores, show_plots)
        
        # Initialize analyzers with enhanced data
        self.data = pd.concat([self.data, scores], axis=1)
        cluster_analyzer = ClusterAnalyzer(scores)
        dimension_analyzer = DimensionAnalyzer(scores, self.spectrum_indicators)
        stats_analyzer = StatsAnalyzer(self.data, self.spectrum_indicators)
        
        # Run analyses
        print("Performing cluster analysis...")
        cluster_results = {
            'optimal_clusters': cluster_analyzer.find_optimal_clusters(),
            'clustering_comparison': cluster_analyzer.compare_clustering_methods()
        }
        
        print("Performing dimensionality analysis...")
        dimension_results = dimension_analyzer.run_dimension_analysis()
        
        print("Performing statistical analysis...")
        stats_results = stats_analyzer.run_statistical_analysis()
        
        # Combine results
        results = {
            'spectrum_scores': scores.to_dict(),
            'distribution_stats': distribution_stats,
            'cluster_analysis': cluster_results,
            'dimension_analysis': dimension_results,
            'statistical_analysis': stats_results
        }
        
        # Generate overall summary report
        summary = self.generate_summary_report(results)
        results['summary'] = summary
        
        print("Political spectrum analysis completed!")
        print(f"Figures saved in: {self.figures_dir}")
        print(f"Reports saved in: {self.reports_dir}")
        
        return results 