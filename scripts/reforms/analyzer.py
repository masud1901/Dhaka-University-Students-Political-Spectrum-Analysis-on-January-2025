"""Reform preferences analysis module for DU student survey."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .utils import setup_plot_style, save_plot


class ReformAnalyzer:
    """Analyzes reform preferences in the survey data."""
    
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
        
        # Reform choices with categories
        self.reform_choices = {
            'quality_education': {
                'category': 'social',
                'label': 'Quality Education',
                'priority': 'high'
            },
            'affordable_healthcare': {
                'category': 'social',
                'label': 'Affordable Healthcare',
                'priority': 'high'
            },
            'better_infrastructure': {
                'category': 'economic',
                'label': 'Better Infrastructure',
                'priority': 'medium'
            },
            'smart_agriculture': {
                'category': 'economic',
                'label': 'Smart Agriculture',
                'priority': 'medium'
            },
            'tech_innovation': {
                'category': 'economic',
                'label': 'Tech Innovation',
                'priority': 'medium'
            },
            'climate_action': {
                'category': 'environmental',
                'label': 'Climate Action',
                'priority': 'high'
            },
            'job_creation': {
                'category': 'economic',
                'label': 'Job Creation',
                'priority': 'high'
            }
        }

    def analyze_popular_reforms(self, show_plots: bool = False):
        """Analyze most popular reform choices."""
        reform_columns = [f'reform_{choice}' for choice in self.reform_choices.keys()]
        
        # Calculate total selections for each reform
        reform_totals = self.data[reform_columns].sum().sort_values(ascending=False)
        total_respondents = len(self.data)
        
        # Calculate percentages
        reform_percentages = (reform_totals / total_respondents) * 100
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=reform_percentages.values, 
                        y=[col.replace('reform_', '').replace('_', ' ').title() 
                           for col in reform_percentages.index])
        
        plt.title('Most Popular Reform Choices')
        plt.xlabel('Percentage of Students')
        plt.ylabel('Reform Type')
        
        # Add percentage labels
        for i, v in enumerate(reform_percentages):
            ax.text(v, i, f'{v:.1f}%', va='center')
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'popular_reforms.png',
            show_plots
        )
        
        return reform_percentages.to_dict()

    def analyze_reform_categories(self, show_plots: bool = False):
        """Analyze reform preferences by category."""
        reform_columns = [f'reform_{choice}' for choice in self.reform_choices.keys()]
        
        # Calculate category preferences
        category_scores = {}
        for category in set(info['category'] for info in self.reform_choices.values()):
            category_cols = [
                f'reform_{choice}' for choice, info in self.reform_choices.items()
                if info['category'] == category
            ]
            category_scores[category] = self.data[category_cols].mean().mean() * 100
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(category_scores.keys(), category_scores.values())
        plt.title('Reform Preferences by Category')
        plt.ylabel('Preference Score (%)')
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        save_plot(plt, self.figures_dir / 'reform_categories.png', show_plots)
        
        return category_scores

    def analyze_reform_combinations(self, show_plots: bool = False):
        """Analyze common combinations of reform choices."""
        reform_columns = [f'reform_{choice}' for choice in self.reform_choices.keys()]
        
        # Get all reform combinations
        reform_patterns = self.data[reform_columns].apply(
            lambda x: tuple(col for col, val in zip(reform_columns, x) if val == 1),
            axis=1
        )
        
        # Count combinations
        pattern_counts = reform_patterns.value_counts().head(5)  # Top 5 combinations
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(pattern_counts)), pattern_counts.values)
        plt.title('Most Common Reform Combinations')
        plt.xlabel('Combination')
        plt.ylabel('Number of Students')
        
        # Format x-axis labels
        plt.xticks(
            range(len(pattern_counts)),
            ['\n'.join([self.reform_choices[c.replace('reform_', '')]['label'] 
                       for c in combo]) for combo in pattern_counts.index],
            rotation=45, ha='right'
        )
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        save_plot(plt, self.figures_dir / 'reform_combinations.png', show_plots)
        
        return pattern_counts.to_dict()

    def analyze_reform_correlations(self, show_plots: bool = False):
        """Analyze correlations between reform choices and other variables."""
        reform_columns = [f'reform_{choice}' for choice in self.reform_choices.keys()]
        
        # 1. Correlation with political change views
        if 'political_change_needed' in self.data.columns:
            change_correlations = {}
            for col in reform_columns:
                corr = self.data[col].corr(self.data['political_change_needed'])
                change_correlations[col] = round(corr, 3)
            
            # Create correlation plot
            plt.figure(figsize=(12, 6))
            bars = plt.bar(
                [self.reform_choices[c.replace('reform_', '')]['label'] 
                 for c in change_correlations.keys()],
                change_correlations.values(),
                color=['lightblue' if v >= 0 else 'lightcoral' 
                       for v in change_correlations.values()]
            )
            plt.title('Reform Preferences vs Political Change Views')
            plt.xlabel('Reform Areas')
            plt.ylabel('Correlation Coefficient')
            plt.xticks(rotation=45, ha='right')
            
            # Add correlation values on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom' if height >= 0 else 'top'
                )
            
            plt.tight_layout()
            save_plot(
                plt,
                self.figures_dir / 'reform_change_correlations.png',
                show_plots
            )
        
        # 2. Analysis by income level
        if 'family_monthly_income' in self.data.columns:
            try:
                # Create income brackets with unique edges
                income_data = self.data['family_monthly_income'].dropna()
                if len(income_data.unique()) >= 4:  # Ensure enough unique values
                    self.data['income_quartile'] = pd.qcut(
                        income_data,
                        q=4,
                        labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                        duplicates='drop'
                    )
                    
                    # Calculate reform preferences by income quartile
                    reform_by_income = (
                        self.data
                        .groupby('income_quartile')[reform_columns]
                        .mean()
                        .round(3)
                    )
                    
                    # Create heatmap
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(
                        reform_by_income,
                        annot=True,
                        fmt='.2f',
                        cmap='YlOrRd',
                        cbar_kws={'label': 'Preference Score'}
                    )
                    plt.title('Reform Preferences by Income Level')
                    plt.xlabel('Reform Areas')
                    plt.ylabel('Income Quartile')
                    
                    plt.tight_layout()
                    save_plot(
                        plt,
                        self.figures_dir / 'reform_income_heatmap.png',
                        show_plots
                    )
                    
                    return {
                        'political_change_correlations': change_correlations,
                        'income_preferences': reform_by_income.to_dict()
                    }
                else:
                    print("Warning: Not enough unique income values for quartile analysis")
                    return {'political_change_correlations': change_correlations}
            except Exception as e:
                print(f"Warning: Error in income analysis: {str(e)}")
                return {'political_change_correlations': change_correlations}
        
        return {'political_change_correlations': change_correlations}

    def analyze_program_differences(self, show_plots: bool = False):
        """Analyze differences in reform preferences between programs."""
        reform_columns = [f'reform_{choice}' for choice in self.reform_choices.keys()]
        
        # Calculate preferences by program
        program_preferences = self.data.groupby('program')[reform_columns].mean()
        
        # Perform statistical tests
        from scipy import stats
        statistical_tests = {}
        for col in reform_columns:
            honours = self.data[self.data['program'] == 'honours'][col]
            masters = self.data[self.data['program'] == 'masters'][col]
            t_stat, p_val = stats.ttest_ind(honours, masters)
            statistical_tests[col] = {'t_statistic': t_stat, 'p_value': p_val}
        
        # Create grouped bar plot
        program_preferences.plot(kind='bar', figsize=(12, 6))
        plt.title('Reform Preferences by Program')
        plt.xlabel('Program')
        plt.ylabel('Proportion of Students')
        plt.legend(
            title='Reform Type',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        plt.tight_layout()
        
        save_plot(plt, self.figures_dir / 'program_preferences.png', show_plots)
        
        return {
            'preferences': program_preferences.to_dict(),
            'statistical_tests': statistical_tests
        }

    def generate_summary_report(self, results: dict):
        """Generate detailed summary report of reform analysis."""
        report = [
            "Reform Preferences Analysis Summary",
            "================================\n"
        ]
        
        # Popular reforms
        if results.get('popular_reforms'):
            report.extend([
                "Most Popular Reforms:",
                "------------------"
            ])
            sorted_reforms = sorted(
                results['popular_reforms'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for reform, percentage in sorted_reforms:
                reform_name = (
                    reform
                    .replace('reform_', '')
                    .replace('_', ' ')
                    .title()
                )
                report.append(f"{reform_name}: {percentage:.1f}%")
        
        # Reform categories
        if results.get('reform_categories'):
            report.extend([
                "\nReform Categories:",
                "-----------------"
            ])
            for category, score in results['reform_categories'].items():
                report.append(f"{category.title()}: {score:.1f}%")
        
        # Common combinations
        if results.get('reform_combinations'):
            report.extend([
                "\nPopular Reform Combinations:",
                "-------------------------"
            ])
            for combo, count in results['reform_combinations'].items():
                reforms = [
                    self.reform_choices[c.replace('reform_', '')]['label']
                    for c in combo
                ]
                report.append(f"{' + '.join(reforms)}: {count} students")
        
        # Program differences
        if results.get('program_differences'):
            report.extend([
                "\nProgram Differences:",
                "------------------"
            ])
            for reform, test in results['program_differences']['statistical_tests'].items():
                reform_name = (
                    reform
                    .replace('reform_', '')
                    .replace('_', ' ')
                    .title()
                )
                sig = "significant" if test['p_value'] < 0.05 else "not significant"
                report.append(
                    f"{reform_name}: {sig} difference "
                    f"(p={test['p_value']:.3f})"
                )
        
        # Add new sections for political change and income analysis
        if results.get('reform_correlations', {}).get('political_change_correlations'):
            report.extend([
                "\nReform Preferences vs Political Change Views:",
                "----------------------------------------"
            ])
            correlations = results['reform_correlations']['political_change_correlations']
            for reform, corr in correlations.items():
                reform_label = self.reform_choices[reform.replace('reform_', '')]['label']
                report.append(
                    f"{reform_label}: {corr:.3f} correlation with change views"
                )
        
        if results.get('reform_correlations', {}).get('income_preferences'):
            report.extend([
                "\nReform Preferences by Income Level:",
                "--------------------------------"
            ])
            income_prefs = results['reform_correlations']['income_preferences']
            for income_level, reforms in income_prefs.items():
                report.append(f"\n{income_level} Income:")
                for reform, score in reforms.items():
                    reform_label = self.reform_choices[reform.replace('reform_', '')]['label']
                    report.append(f"  {reform_label}: {score:.2f}")
        
        # Save report
        report_path = self.reports_dir / 'reform_summary.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def run_reform_analysis(self, show_plots: bool = False) -> dict:
        """Run complete reform analysis."""
        print("Running reform preferences analysis...")
        
        results = {
            'popular_reforms': self.analyze_popular_reforms(show_plots),
            'reform_categories': self.analyze_reform_categories(show_plots),
            'reform_combinations': self.analyze_reform_combinations(show_plots),
            'reform_correlations': self.analyze_reform_correlations(show_plots),
            'program_differences': self.analyze_program_differences(show_plots)
        }
        
        # Generate summary report
        summary = self.generate_summary_report(results)
        results['summary'] = summary
        
        print("Reform analysis completed!")
        print(f"Figures saved in: {self.figures_dir}")
        print(f"Reports saved in: {self.reports_dir}")
        
        return results 