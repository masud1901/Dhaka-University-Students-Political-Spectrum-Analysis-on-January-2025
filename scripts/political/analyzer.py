"""Political spectrum analysis module for DU student survey."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .utils import setup_plot_style, save_plot
import numpy as np


class PoliticalAnalyzer:
    """Analyzes political spectrum aspects of the survey data."""
    
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
        
        # Binary choice columns with their labels and dimensions
        self.binary_columns = {
            'economic_growth_over_environment': {
                'label': 'Economic Growth vs Environment',
                'dimension': 'economic',
                'high_label': 'Growth-focused',
                'low_label': 'Environment-focused'
            },
            'individual_freedom_over_harmony': {
                'label': 'Individual Freedom vs Social Harmony',
                'dimension': 'social',
                'high_label': 'Individualistic',
                'low_label': 'Collectivist'
            },
            'traditional_over_progressive': {
                'label': 'Traditional vs Progressive Values',
                'dimension': 'cultural',
                'high_label': 'Traditional',
                'low_label': 'Progressive'
            },
            'central_over_local': {
                'label': 'Central vs Local Authority',
                'dimension': 'governance',
                'high_label': 'Centralist',
                'low_label': 'Localist'
            },
            'market_over_regulation': {
                'label': 'Market Freedom vs Regulation',
                'dimension': 'economic',
                'high_label': 'Free Market',
                'low_label': 'Regulatory'
            },
            'stability_over_justice': {
                'label': 'Stability vs Justice',
                'dimension': 'social',
                'high_label': 'Stability-focused',
                'low_label': 'Justice-focused'
            }
        }

    def analyze_binary_choices(self, show_plots: bool = False):
        """Analyze binary choice preferences."""
        results = {}
        
        # Calculate preferences for each binary choice
        for col, label in self.binary_columns.items():
            if col not in self.data.columns:
                continue
                
            # Calculate proportions
            proportions = self.data[col].value_counts(normalize=True) * 100
            results[col] = proportions.to_dict()
            
            # Create plot
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=['First Choice', 'Second Choice'], 
                           y=[proportions.get(1, 0), proportions.get(0, 0)])
            plt.title(f'Preference Distribution: {label}')
            plt.ylabel('Percentage of Respondents')
            
            # Add percentage labels
            for i, v in enumerate([proportions.get(1, 0), proportions.get(0, 0)]):
                ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
            
            # Save plot
            save_plot(
                plt,
                self.figures_dir / f'{col}_distribution.png',
                show_plots
            )
        
        return results

    def analyze_binary_choices_by_program(self, show_plots: bool = False):
        """Analyze binary choice preferences by program."""
        results = {}
        
        for col, info in self.binary_columns.items():
            if col not in self.data.columns:
                continue
                
            # Calculate proportions by program
            props = (
                self.data
                .groupby('program')[col]
                .agg(['mean', 'count'])
                .round(3)
            )
            results[col] = props.to_dict('index')
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.data, x='program', y=col)
            plt.title(f'{info["label"]} Preference by Program')
            plt.ylabel('Proportion Choosing First Option')
            
            # Add percentage labels
            for i, prog in enumerate(props.index):
                pct = props.loc[prog, 'mean'] * 100
                plt.text(i, props.loc[prog, 'mean'], 
                        f'{pct:.1f}%', ha='center', va='bottom')
            
            save_plot(plt, self.figures_dir / f'{col}_by_program.png', show_plots)
            
        return results

    def analyze_political_dimensions(self, show_plots: bool = False):
        """Analyze combined political dimensions."""
        # Group columns by dimension
        dimensions = {}
        for col, info in self.binary_columns.items():
            dim = info['dimension']
            if dim not in dimensions:
                dimensions[dim] = []
            dimensions[dim].append(col)
        
        # Calculate dimension scores
        scores = {}
        for dim, cols in dimensions.items():
            scores[dim] = self.data[cols].mean(axis=1).mean() * 100
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False)
        values = [scores[dim] for dim in dimensions]
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(dimensions.keys())
        plt.title('Political Dimension Scores')
        
        save_plot(plt, self.figures_dir / 'political_dimensions.png', show_plots)
        
        return scores

    def analyze_ideological_clusters(self, show_plots: bool = False):
        """Analyze ideological clustering of respondents."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for clustering
        binary_cols = list(self.binary_columns.keys())
        X = self.data[binary_cols].copy()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        self.data['ideology_cluster'] = clusters
        
        # Calculate cluster characteristics
        cluster_profiles = (
            self.data
            .groupby('ideology_cluster')[binary_cols]
            .mean()
            .round(3)
        )
        
        # Create cluster visualization
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=self.data,
            x='market_over_regulation',
            y='individual_freedom_over_harmony',
            hue='ideology_cluster',
            style='program'
        )
        plt.title('Ideological Clusters')
        plt.xlabel('Economic Dimension (Market → Regulation)')
        plt.ylabel('Social Dimension (Individual → Collective)')
        
        save_plot(plt, self.figures_dir / 'ideology_clusters.png', show_plots)
        
        return cluster_profiles.to_dict('index')

    def analyze_program_differences(self, show_plots: bool = False):
        """Analyze political differences between programs."""
        binary_cols = list(self.binary_columns.keys())
        
        # Calculate mean scores by program
        program_scores = (
            self.data
            .groupby('program')[binary_cols]
            .agg(['mean', 'std'])
            .round(3)
        )
        
        # Perform t-tests between programs
        from scipy import stats
        t_tests = {}
        for col in binary_cols:
            prog1 = self.data[self.data['program'] == 'honours'][col]
            prog2 = self.data[self.data['program'] == 'masters'][col]
            t_stat, p_val = stats.ttest_ind(prog1, prog2)
            t_tests[col] = {'t_statistic': t_stat, 'p_value': p_val}
        
        return {
            'program_scores': program_scores.to_dict(),
            'statistical_tests': t_tests
        }

    def analyze_political_change_views(self, show_plots: bool = False):
        """Analyze views on political change needs."""
        if 'political_change_needed' not in self.data.columns:
            print("Warning: Political change data not found")
            return {}
            
        # Calculate overall distribution
        change_dist = (
            self.data['political_change_needed']
            .value_counts(normalize=True) * 100
        )
        
        # Create mapping for labels
        label_map = {
            1: 'Change Needed',
            0: 'No Change Needed',
            pd.NA: 'Uncertain'  # Handle NA values explicitly
        }
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Prepare data for pie chart
        plot_data = {}
        for val in change_dist.index:
            key = label_map.get(val, 'Other')
            plot_data[key] = change_dist[val]
        
        # Create pie chart
        plt.pie(
            plot_data.values(),
            labels=plot_data.keys(),
            autopct='%1.1f%%',
            colors=['lightcoral', 'lightblue', 'lightgray']
        )
        plt.title('Views on Political Change Needs')
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'political_change_views.png',
            show_plots
        )
        
        return plot_data

    def analyze_change_views_relationships(self, show_plots: bool = False):
        """Analyze relationships between political change views and other factors."""
        if 'political_change_needed' not in self.data.columns:
            return {}
            
        results = {}
        
        # 1. Analysis by program
        views_by_program = pd.crosstab(
            self.data['program'],
            self.data['political_change_needed'],
            normalize='index'
        ) * 100
        
        results['by_program'] = views_by_program.to_dict()
        
        # Create stacked bar plot for program distribution
        plt.figure(figsize=(10, 6))
        views_by_program.plot(
            kind='bar',
            stacked=True,
            color=['lightblue', 'lightcoral', 'lightgray']
        )
        plt.title('Political Change Views by Program')
        plt.xlabel('Program')
        plt.ylabel('Percentage')
        plt.legend(['No Change Needed', 'Change Needed', 'Uncertain'])
        plt.xticks(rotation=45)
        
        # Save program plot
        save_plot(
            plt,
            self.figures_dir / 'change_views_by_program.png',
            show_plots
        )
        
        # 2. Analysis by income level (if available)
        if 'family_monthly_income' in self.data.columns:
            try:
                # Create income brackets with handling for duplicates
                income_data = self.data['family_monthly_income'].dropna()
                if len(income_data.unique()) >= 4:  # Ensure enough unique values
                    self.data['income_bracket'] = pd.qcut(
                        income_data,
                        q=4,
                        labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                        duplicates='drop'
                    )
                    
                    views_by_income = pd.crosstab(
                        self.data['income_bracket'],
                        self.data['political_change_needed'],
                        normalize='index'
                    ) * 100
                    
                    results['by_income'] = views_by_income.to_dict()
                    
                    # Create stacked bar plot for income distribution
                    plt.figure(figsize=(10, 6))
                    views_by_income.plot(
                        kind='bar',
                        stacked=True,
                        color=['lightblue', 'lightcoral', 'lightgray']
                    )
                    plt.title('Political Change Views by Income Level')
                    plt.xlabel('Income Level')
                    plt.ylabel('Percentage')
                    plt.legend(['No Change Needed', 'Change Needed', 'Uncertain'])
                    plt.xticks(rotation=45)
                    
                    # Save income plot
                    save_plot(
                        plt,
                        self.figures_dir / 'change_views_by_income.png',
                        show_plots
                    )
                else:
                    print("Warning: Not enough unique income values for quartile analysis")
            except Exception as e:
                print(f"Warning: Income analysis failed: {str(e)}")
        
        # 3. Correlation analysis with other variables
        correlations = {}
        for col in self.binary_columns:
            if col in self.data.columns:
                corr = self.data[col].corr(self.data['political_change_needed'])
                correlations[col] = round(corr, 3)
        
        results['correlations'] = correlations
        
        # Create correlation plot
        if correlations:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(
                range(len(correlations)),
                correlations.values(),
                color=['lightblue' if v >= 0 else 'lightcoral' 
                       for v in correlations.values()]
            )
            plt.title('Correlations with Political Change Views')
            plt.xlabel('Variables')
            plt.ylabel('Correlation Coefficient')
            plt.xticks(
                range(len(correlations)),
                [self.binary_columns[col]['label'] for col in correlations.keys()],
                rotation=45,
                ha='right'
            )
            
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
            
            # Save correlation plot
            save_plot(
                plt,
                self.figures_dir / 'change_views_correlations.png',
                show_plots
            )
        
        return results

    def generate_summary_report(self, results: dict):
        """Generate detailed summary report of political analysis."""
        report = [
            "Political Analysis Summary",
            "========================\n"
        ]
        
        # Binary choices summary
        if results.get('binary_choices'):
            report.append("Binary Choice Preferences by Program:")
            report.append("--------------------------------")
            for col, prog_data in results['binary_choices'].items():
                info = self.binary_columns[col]
                report.append(f"\n{info['label']}:")
                for prog, stats in prog_data.items():
                    pct = stats['mean'] * 100
                    n = stats['count']
                    report.append(
                        f"{prog}: {pct:.1f}% chose {info['high_label']} (n={n})"
                    )
        
        # Political dimensions
        if results.get('political_dimensions'):
            report.extend([
                "\nPolitical Dimension Scores:",
                "------------------------"
            ])
            for dim, score in results['political_dimensions'].items():
                report.append(f"{dim.title()}: {score:.1f}")
        
        # Ideological clusters
        if results.get('ideological_clusters'):
            report.extend([
                "\nIdeological Clusters:",
                "-------------------"
            ])
            for cluster, profile in results['ideological_clusters'].items():
                report.append(f"\nCluster {cluster}:")
                for measure, value in profile.items():
                    info = self.binary_columns[measure]
                    tendency = info['high_label'] if value > 0.5 else info['low_label']
                    strength = abs(value - 0.5) * 2  # Convert to 0-1 scale
                    report.append(
                        f"  {info['label']}: {tendency} "
                        f"(strength: {strength:.2f})"
                    )
        
        # Program differences
        if results.get('program_differences'):
            report.extend([
                "\nProgram Differences:",
                "------------------"
            ])
            for col, test in results['program_differences']['statistical_tests'].items():
                info = self.binary_columns[col]
                sig = "significant" if test['p_value'] < 0.05 else "not significant"
                report.append(
                    f"{info['label']}: {sig} difference "
                    f"(p={test['p_value']:.3f})"
                )
        
        # Political change views
        if results.get('political_change_views'):
            report.extend([
                "\nPolitical Change Views:",
                "-------------------"
            ])
            for view, percentage in results['political_change_views'].items():
                report.append(f"{view}: {percentage:.1f}%")
        
        # Change views relationships
        if results.get('change_views_relationships'):
            report.extend([
                "\nChange Views Relationships:",
                "-------------------"
            ])
            for col, corr in results['change_views_relationships']['correlations'].items():
                report.append(f"{col}: {corr}")
        
        # Save report
        report_path = self.reports_dir / 'political_summary.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def run_political_analysis(self, show_plots: bool = False) -> dict:
        """Run complete political analysis."""
        print("Running political spectrum analysis...")
        
        results = {
            'binary_choices': self.analyze_binary_choices_by_program(show_plots),
            'political_dimensions': self.analyze_political_dimensions(show_plots),
            'ideological_clusters': self.analyze_ideological_clusters(show_plots),
            'program_differences': self.analyze_program_differences(show_plots),
            'political_change_views': self.analyze_political_change_views(show_plots),
            'change_views_relationships': self.analyze_change_views_relationships(show_plots)
        }
        
        # Generate summary report
        summary = self.generate_summary_report(results)
        results['summary'] = summary
        
        print("Political analysis completed!")
        print(f"Figures saved in: {self.figures_dir}")
        print(f"Reports saved in: {self.reports_dir}")
        
        return results 