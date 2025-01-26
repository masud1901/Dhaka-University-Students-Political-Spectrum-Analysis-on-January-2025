"""Advanced statistical analysis module for political spectrum."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from .utils import setup_plot_style, save_plot


class StatsAnalyzer:
    """Performs advanced statistical analysis on political data."""
    
    def __init__(self, data: pd.DataFrame, indicators: dict):
        """Initialize with full dataset and indicator config."""
        self.data = data
        self.indicators = indicators
        self.output_dir = Path(__file__).parent
        self.figures_dir = self.output_dir / 'figures' / 'stats'
        self.reports_dir = self.output_dir / 'reports'
        
        # Create output directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting style
        setup_plot_style()

    def analyze_program_differences(self):
        """Analyze differences between programs."""
        results = {}
        
        for dimension, config in self.indicators.items():
            # Get scores for each program
            honours = self.data[self.data['program'] == 'honours'][f'{dimension}_spectrum']
            masters = self.data[self.data['program'] == 'masters'][f'{dimension}_spectrum']
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(honours, masters)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((honours.var() + masters.var()) / 2)
            cohens_d = (honours.mean() - masters.mean()) / pooled_std
            
            results[dimension] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'effect_size': cohens_d,
                'means': {
                    'honours': honours.mean(),
                    'masters': masters.mean()
                }
            }
            
        return results
        
    def analyze_value_relationships(self):
        """Analyze relationships with other value measures."""
        value_columns = [
            'individualism_score',
            'tradition_reliance_score',
            'cultural_importance_score',
            'leadership_satisfaction_score'
        ]
        
        correlations = {}
        for dimension in self.indicators.keys():
            spectrum_score = f'{dimension}_spectrum'
            if spectrum_score in self.data.columns:
                correlations[dimension] = {}
                for value_col in value_columns:
                    if value_col in self.data.columns:
                        corr = self.data[spectrum_score].corr(self.data[value_col])
                        correlations[dimension][value_col] = corr
                        
        return correlations
        
    def perform_regression_analysis(self):
        """Perform multiple regression analysis."""
        regression_results = {}
        
        # For each political dimension
        for dimension in ['economic', 'social', 'cultural']:
            target = f'{dimension}_score'
            if target not in self.data.columns:
                continue
                
            # Get predictors
            predictors = self.indicators.get(f'{dimension}_features', [])
            if not predictors:
                continue
                
            # Prepare data
            X = self.data[predictors].fillna(0)
            y = self.data[target]
            
            # Add constant
            X = add_constant(X)
            
            try:
                # Fit regression
                model = OLS(y, X).fit()
                
                # Store results
                regression_results[dimension] = {
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'coefficients': model.params.to_dict(),
                    'p_values': model.pvalues.to_dict(),
                    'summary': model.summary().as_text()
                }
                
                # Plot actual vs predicted
                plt.figure(figsize=(10, 6))
                plt.scatter(y, model.fittedvalues, alpha=0.5)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title(f'{dimension.title()} Score: Actual vs Predicted')
                
                # Save plot
                save_plot(
                    plt,
                    self.figures_dir / f'{dimension}_regression.png',
                    show_plots=False
                )
                
            except Exception as e:
                print(f"Warning: Regression failed for {dimension}: {str(e)}")
                continue
        
        return regression_results

    def perform_anova_analysis(self):
        """Perform ANOVA and post-hoc analysis."""
        anova_results = {}
        
        # For each score dimension
        for dimension in ['economic', 'social', 'cultural']:
            score_col = f'{dimension}_score'
            if score_col not in self.data.columns:
                continue
            
            # Perform one-way ANOVA by program
            try:
                programs = self.data['program'].unique()
                scores_by_program = [
                    self.data[self.data['program'] == prog][score_col]
                    for prog in programs
                ]
                
                # ANOVA
                f_stat, p_val = stats.f_oneway(*scores_by_program)
                
                # Tukey's HSD
                scores = self.data[score_col]
                groups = self.data['program']
                tukey = pairwise_tukeyhsd(scores, groups)
                
                # Store results
                anova_results[dimension] = {
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'tukey_results': str(tukey),
                    'mean_by_program': self.data.groupby('program')[score_col]
                        .mean()
                        .to_dict()
                }
                
                # Plot boxplot
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='program', y=score_col, data=self.data)
                plt.title(f'{dimension.title()} Scores by Program')
                plt.xticks(rotation=45)
                
                # Add ANOVA results to plot
                plt.text(
                    0.02, 0.98,
                    f'F-stat: {f_stat:.2f}\np-value: {p_val:.4f}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                
                # Save plot
                save_plot(
                    plt,
                    self.figures_dir / f'{dimension}_anova.png',
                    show_plots=False
                )
                
            except Exception as e:
                print(f"Warning: ANOVA failed for {dimension}: {str(e)}")
                continue
        
        return anova_results

    def perform_markov_analysis(self):
        """Perform Markov chain analysis on preference transitions."""
        # Define preference states
        states = ['traditional', 'moderate', 'progressive']
        n_states = len(states)
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_states, n_states))
        
        try:
            # Calculate transitions based on multiple preference indicators
            for col in ['traditional_over_progressive', 'stability_over_justice']:
                if col not in self.data.columns:
                    continue
                    
                # Convert preferences to state indices with handling for duplicates
                try:
                    state_indices = pd.qcut(
                        self.data[col],
                        q=n_states,
                        labels=range(n_states),
                        duplicates='drop'  # Handle duplicate bin edges
                    )
                except ValueError as e:
                    print(f"Warning: Could not create quantiles for {col}: {str(e)}")
                    # Try alternative binning if data is too discrete
                    unique_vals = self.data[col].nunique()
                    if unique_vals < n_states:
                        # Use actual values if too few unique values
                        bins = min(unique_vals, n_states)
                        state_indices = pd.qcut(
                            self.data[col].rank(method='first'),  # Use ranks
                            q=bins,
                            labels=range(bins),
                            duplicates='drop'
                        )
                    else:
                        continue
                
                # Count transitions
                valid_indices = state_indices.dropna()
                for i in range(len(valid_indices)-1):
                    from_state = valid_indices.iloc[i]
                    to_state = valid_indices.iloc[i+1]
                    if pd.notna(from_state) and pd.notna(to_state):
                        transition_matrix[int(from_state)][int(to_state)] += 1
            
            # Check if we have any transitions
            if not np.any(transition_matrix):
                print("Warning: No valid transitions found for Markov analysis")
                return {}
            
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(
                transition_matrix,
                row_sums[:, np.newaxis],
                where=row_sums[:, np.newaxis] != 0,
                out=np.zeros_like(transition_matrix)
            )
            
            # Plot transition matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                transition_matrix,
                annot=True,
                fmt='.2f',
                xticklabels=states[:transition_matrix.shape[1]],
                yticklabels=states[:transition_matrix.shape[0]],
                cmap='YlOrRd'
            )
            plt.title('Preference Transition Probabilities')
            plt.xlabel('To State')
            plt.ylabel('From State')
            
            # Save plot
            save_plot(
                plt,
                self.figures_dir / 'markov_transitions.png',
                show_plots=False
            )
            
            return {
                'transition_matrix': transition_matrix.tolist(),
                'states': states[:transition_matrix.shape[0]]
            }
            
        except Exception as e:
            print(f"Warning: Markov analysis failed: {str(e)}")
            return {}

    def analyze_income_effects(self):
        """Analyze the effects of income on political dimensions."""
        if 'family_monthly_income' not in self.data.columns:
            return {}
            
        results = {}
        
        try:
            # Create income quartiles
            self.data['income_quartile'] = pd.qcut(
                self.data['family_monthly_income'].dropna(),
                q=4,
                labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                duplicates='drop'
            )
            
            # Analyze each dimension
            for dimension in self.indicators.keys():
                spectrum_col = f'{dimension}_spectrum'
                if spectrum_col not in self.data.columns:
                    continue
                    
                # ANOVA by income quartile
                groups = [
                    self.data[self.data['income_quartile'] == q][spectrum_col]
                    for q in self.data['income_quartile'].unique()
                ]
                f_stat, p_val = stats.f_oneway(*groups)
                
                results[dimension] = {
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'means_by_quartile': (
                        self.data
                        .groupby('income_quartile')[spectrum_col]
                        .mean()
                        .to_dict()
                    )
                }
                
        except Exception as e:
            print(f"Warning: Income analysis failed: {str(e)}")
            
        return results

    def analyze_political_change_effects(self):
        """Analyze effects of political change views."""
        if 'political_change_needed' not in self.data.columns:
            return {}
            
        results = {}
        
        for dimension in self.indicators.keys():
            spectrum_col = f'{dimension}_spectrum'
            if spectrum_col not in self.data.columns:
                continue
                
            # T-test between change/no-change groups
            change = self.data[self.data['political_change_needed'] == 1][spectrum_col]
            no_change = self.data[self.data['political_change_needed'] == 0][spectrum_col]
            
            if len(change) > 0 and len(no_change) > 0:
                t_stat, p_val = stats.ttest_ind(change, no_change)
                
                results[dimension] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'mean_change': change.mean(),
                    'mean_no_change': no_change.mean()
                }
                
        return results

    def generate_stats_report(self, results: dict):
        """Generate statistical analysis report."""
        report = [
            "Advanced Statistical Analysis Report",
            "================================\n"
        ]
        
        # Regression results
        if results.get('regression'):
            report.extend([
                "Regression Analysis:",
                "------------------"
            ])
            for dim, res in results['regression'].items():
                report.extend([
                    f"\n{dim.title()} Dimension:",
                    f"R-squared: {res['r_squared']:.3f}",
                    f"Adjusted R-squared: {res['adj_r_squared']:.3f}",
                    "\nSignificant predictors:"
                ])
                for pred, p_val in res['p_values'].items():
                    if p_val < 0.05:
                        coef = res['coefficients'][pred]
                        report.append(
                            f"{pred}: {coef:.3f} (p={p_val:.3f})"
                        )
        
        # ANOVA results
        if results.get('anova'):
            report.extend([
                "\nANOVA Analysis:",
                "--------------"
            ])
            for dim, res in results['anova'].items():
                report.extend([
                    f"\n{dim.title()} Dimension:",
                    f"F-statistic: {res['f_statistic']:.3f}",
                    f"p-value: {res['p_value']:.3f}"
                ])
        
        # Markov analysis
        if results.get('markov'):
            report.extend([
                "\nMarkov Chain Analysis:",
                "-------------------",
                "See transition matrix plot for details."
            ])
        
        # Income effects
        if results.get('income_effects'):
            report.extend([
                "\nIncome Effects Analysis:",
                "---------------------"
            ])
            for dim, res in results['income_effects'].items():
                report.extend([
                    f"\n{dim.title()} Dimension:",
                    f"F-statistic: {res['f_statistic']:.3f}",
                    f"p-value: {res['p_value']:.3f}",
                    "\nMeans by Income Quartile:"
                ])
                for quartile, mean in res['means_by_quartile'].items():
                    report.append(f"{quartile}: {mean:.2f}")
        
        # Political change effects
        if results.get('political_change_effects'):
            report.extend([
                "\nPolitical Change Effects Analysis:",
                "-------------------------"
            ])
            for dim, res in results['political_change_effects'].items():
                report.extend([
                    f"\n{dim.title()} Dimension:",
                    f"T-statistic: {res['t_statistic']:.3f}",
                    f"p-value: {res['p_value']:.3f}",
                    f"Mean Change: {res['mean_change']:.2f}",
                    f"Mean No Change: {res['mean_no_change']:.2f}"
                ])
        
        # Save report
        report_path = self.reports_dir / 'statistical_analysis.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def run_statistical_analysis(self):
        """Run complete statistical analysis."""
        print("Running advanced statistical analysis...")
        
        results = {
            'program_differences': self.analyze_program_differences(),
            'value_relationships': self.analyze_value_relationships(),
            'regression': self.perform_regression_analysis(),
            'anova': self.perform_anova_analysis(),
            'markov': self.perform_markov_analysis(),
            'income_effects': self.analyze_income_effects(),
            'political_change_effects': self.analyze_political_change_effects()
        }
        
        # Generate summary
        summary = self.generate_stats_report(results)
        results['summary'] = summary
        
        print("Statistical analysis completed!")
        print(f"Figures saved in: {self.figures_dir}")
        print(f"Reports saved in: {self.reports_dir}")
        
        return results 