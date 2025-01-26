"""Value analysis module for DU student survey."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .utils import setup_plot_style, save_plot


class ValueAnalyzer:
    """Analyzes value preferences in the survey data."""
    
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
        
        # Value columns with metadata
        self.value_columns = {
            'economic_freedom_score': {
                'label': 'Economic Freedom',
                'dimension': 'economic',
                'description': 'Views on market freedom vs regulation'
            },
            'cultural_importance_score': {
                'label': 'Cultural Values',
                'dimension': 'social',
                'description': 'Importance of cultural traditions'
            },
            'tradition_reliance_score': {
                'label': 'Traditional Values',
                'dimension': 'social',
                'description': 'Reliance on traditional wisdom'
            },
            'leadership_satisfaction_score': {
                'label': 'Leadership',
                'dimension': 'political',
                'description': 'Satisfaction with current leadership'
            },
            'protest_support_score': {
                'label': 'Justice & Equity',
                'dimension': 'political',
                'description': 'Support for justice-oriented actions'
            },
            'individualism_score': {
                'label': 'Individualism',
                'dimension': 'social',
                'description': 'Individual vs collective orientation'
            }
        }

    def analyze_value_importance(self, show_plots: bool = False):
        """Analyze importance of different values."""
        # Calculate mean scores for each value
        value_means = self.data[list(self.value_columns.keys())].mean()
        value_means = value_means.dropna()  # Remove any NaN values
        
        if value_means.empty:
            print("Warning: No valid value scores found")
            return {}
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            y=[self.value_columns[col]['label'] for col in value_means.index],
            x=value_means.values,
            orient='h'  # Horizontal bars
        )
        
        plt.title('Average Value Scores')
        plt.xlabel('Average Score (1-5)')
        plt.ylabel('Value Type')
        
        # Add score labels
        for i, v in enumerate(value_means):
            if np.isfinite(v):  # Only add label if value is finite
                ax.text(v, i, f'{v:.2f}', va='center')
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'value_importance.png',
            show_plots
        )
        
        return value_means.to_dict()

    def analyze_value_distribution_by_program(self, show_plots: bool = False):
        """Analyze value distribution across programs."""
        # Calculate mean scores by program
        value_cols = [col for col in self.value_columns.keys() 
                     if col in self.data.columns]
        
        if not value_cols:
            print("Warning: No value columns found in data")
            return {}
        
        program_values = (
            self.data
            .groupby('program')[value_cols]
            .mean()
            .dropna(how='all')  # Remove rows with all NaN
        )
        
        if program_values.empty:
            print("Warning: No valid program values found")
            return {}
        
        # Create plot
        plt.figure(figsize=(12, 6))
        program_values.plot(kind='bar')
        plt.title('Value Scores by Program')
        plt.xlabel('Program')
        plt.ylabel('Average Score')
        plt.legend(
            [self.value_columns[col]['label'] for col in program_values.columns],
            title='Values',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        plt.xticks(rotation=45)
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'value_by_program.png',
            show_plots
        )
        
        return program_values.to_dict()

    def analyze_value_correlations(self, show_plots: bool = False):
        """Analyze correlations between values and preferences."""
        # Select columns for correlation analysis
        correlation_columns = [
            col for col in [
                'economic_freedom_score',
                'cultural_importance_score',
                'tradition_reliance_score',
                'leadership_satisfaction_score',
                'protest_support_score',
                'economic_growth_over_environment',
                'individual_freedom_over_harmony',
                'market_over_regulation'
            ] if col in self.data.columns
        ]
        
        if not correlation_columns:
            print("Warning: No correlation columns found")
            return {}
        
        # Calculate correlation matrix
        corr_matrix = (
            self.data[correlation_columns]
            .corr()
            .fillna(0)  # Replace NaN with 0 for visualization
        )
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix), k=1)  # Mask upper triangle
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu',
            center=0,
            square=True,
            mask=mask,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Value and Preference Correlations')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'value_correlations.png',
            show_plots
        )
        
        return corr_matrix.to_dict()

    def analyze_key_relationships(self, show_plots: bool = False):
        """Analyze specific value relationships of interest."""
        relationships = [
            {
                'x': 'economic_freedom_score',
                'y': 'market_over_regulation',
                'title': 'Economic Views vs Market Regulation',
                'xlabel': 'Economic Freedom Score',
                'ylabel': 'Market Freedom Preference'
            },
            {
                'x': 'tradition_reliance_score',
                'y': 'cultural_importance_score',
                'title': 'Traditional Values vs Cultural Importance',
                'xlabel': 'Traditional Values Score',
                'ylabel': 'Cultural Importance Score'
            },
            {
                'x': 'leadership_satisfaction_score',
                'y': 'num_reforms_selected',
                'title': 'Leadership Satisfaction vs Reform Interest',
                'xlabel': 'Leadership Satisfaction Score',
                'ylabel': 'Number of Reforms Selected'
            }
        ]
        
        results = {}
        
        for rel in relationships:
            if rel['x'] not in self.data.columns or rel['y'] not in self.data.columns:
                print(f"Warning: Missing columns for {rel['title']}")
                continue
            
            # Create scatter plot with regression line
            plt.figure(figsize=(10, 6))
            sns.regplot(
                data=self.data,
                x=rel['x'],
                y=rel['y'],
                scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'}
            )
            plt.title(rel['title'])
            plt.xlabel(rel['xlabel'])
            plt.ylabel(rel['ylabel'])
            
            # Calculate correlation
            correlation = self.data[rel['x']].corr(self.data[rel['y']])
            plt.text(
                0.05, 0.95,
                f'Correlation: {correlation:.2f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            # Save plot
            filename = (
                f"{rel['title'].lower().replace(' ', '_')}.png"
            )
            save_plot(
                plt,
                self.figures_dir / filename,
                show_plots
            )
            
            # Store results
            results[rel['title']] = {
                'correlation': correlation,
                'mean_x': self.data[rel['x']].mean(),
                'mean_y': self.data[rel['y']].mean()
            }
        
        return results

    def analyze_value_dimensions(self, show_plots: bool = False):
        """Analyze value dimensions and their relationships."""
        # Define value dimensions
        dimensions = {
            'economic': ['economic_freedom_score', 'market_over_regulation'],
            'social': ['individualism_score', 'cultural_importance_score'],
            'political': ['leadership_satisfaction_score', 'political_change_needed'],
            'traditional': ['tradition_reliance_score']
        }
        
        results = {}
        
        # Calculate dimension scores
        for dim, cols in dimensions.items():
            available_cols = [col for col in cols if col in self.data.columns]
            if available_cols:
                results[dim] = self.data[available_cols].mean(axis=1).mean()
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(results), endpoint=False)
        values = list(results.values())
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(results.keys())
        plt.title('Value Dimensions')
        
        save_plot(plt, self.figures_dir / 'value_dimensions.png', show_plots)
        
        return results

    def analyze_value_clusters(self, show_plots: bool = False):
        """Analyze value-based clusters of respondents."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Prepare data for clustering
        value_cols = [col for col in self.value_columns.keys() 
                     if col in self.data.columns]
        
        if len(value_cols) < 2:
            print("Warning: Not enough value columns for clustering")
            return {}
            
        X = self.data[value_cols].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        try:
            # Perform clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to data
            self.data['value_cluster'] = clusters
            
            # Calculate cluster profiles
            cluster_profiles = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_),
                columns=value_cols
            ).round(2)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Select two most distinctive features for visualization
            feature_importance = np.abs(cluster_profiles.std())
            top_features = feature_importance.nlargest(2).index
            
            sns.scatterplot(
                data=self.data,
                x=top_features[0],
                y=top_features[1],
                hue='value_cluster',
                style='program',
                alpha=0.6
            )
            plt.title('Value-Based Clusters')
            plt.xlabel(self.value_columns[top_features[0]]['label'])
            plt.ylabel(self.value_columns[top_features[1]]['label'])
            
            save_plot(plt, self.figures_dir / 'value_clusters.png', show_plots)
            
            # Create detailed cluster profiles
            cluster_summary = {}
            for i in range(4):
                cluster_data = self.data[self.data['value_cluster'] == i]
                cluster_summary[f'Cluster_{i}'] = {
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(self.data)) * 100,
                    'values': cluster_profiles.iloc[i].to_dict(),
                    'program_distribution': cluster_data['program'].value_counts(normalize=True).to_dict()
                }
            
            return cluster_summary
            
        except Exception as e:
            print(f"Warning: Error in clustering analysis: {str(e)}")
            return {}

    def analyze_value_trends(self, show_plots: bool = False):
        """Analyze value trends across sessions."""
        value_cols = [col for col in self.value_columns.keys() 
                     if col in self.data.columns]
        
        # Calculate trends by session
        trends = (
            self.data
            .groupby('session')[value_cols]
            .mean()
            .round(2)
        )
        
        # Create line plot
        plt.figure(figsize=(12, 6))
        for col in value_cols:
            plt.plot(
                trends.index, 
                trends[col], 
                marker='o', 
                label=self.value_columns[col]['label']
            )
        
        plt.title('Value Trends Across Sessions')
        plt.xlabel('Session')
        plt.ylabel('Average Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_plot(plt, self.figures_dir / 'value_trends.png', show_plots)
        
        return trends.to_dict()

    def analyze_value_interactions(self, show_plots: bool = False):
        """Analyze interactions between different values."""
        value_cols = [col for col in self.value_columns.keys() 
                     if col in self.data.columns]
        
        if len(value_cols) < 2:
            return {}
        
        interactions = []
        for i, col1 in enumerate(value_cols):
            for col2 in value_cols[i+1:]:
                # Calculate correlation
                corr = self.data[col1].corr(self.data[col2])
                
                # Perform chi-square test for independence
                from scipy.stats import chi2_contingency
                contingency = pd.crosstab(
                    self.data[col1].round(),
                    self.data[col2].round()
                )
                chi2, p_val, _, _ = chi2_contingency(contingency)
                
                interactions.append({
                    'value1': self.value_columns[col1]['label'],
                    'value2': self.value_columns[col2]['label'],
                    'correlation': corr,
                    'chi2_stat': chi2,
                    'p_value': p_val
                })
        
        # Create network visualization
        if show_plots and interactions:
            import networkx as nx
            G = nx.Graph()
            
            # Add nodes
            for col in value_cols:
                G.add_node(self.value_columns[col]['label'])
            
            # Add edges
            for interaction in interactions:
                if abs(interaction['correlation']) > 0.2:  # Only show stronger correlations
                    G.add_edge(
                        interaction['value1'],
                        interaction['value2'],
                        weight=abs(interaction['correlation'])
                    )
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(
                G, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=10,
                font_weight='bold'
            )
            
            save_plot(plt, self.figures_dir / 'value_network.png', show_plots)
        
        return interactions

    def analyze_income_value_relationships(self, show_plots: bool = False):
        """Analyze relationships between income levels and values."""
        if 'family_monthly_income' not in self.data.columns:
            return {}
            
        try:
            # Create income brackets with unique edges
            income_data = self.data['family_monthly_income'].dropna()
            if len(income_data.unique()) < 4:
                print("Warning: Not enough unique income values for quartile analysis")
                return {}
                
            # Create income quartiles with duplicate handling
            self.data['income_quartile'] = pd.qcut(
                income_data,
                q=4,
                labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                duplicates='drop'
            )
            
            # Define value columns to analyze
            value_cols = [
                'economic_freedom_score',
                'individualism_score',
                'cultural_importance_score',
                'leadership_satisfaction_score',
                'tradition_reliance_score',
                'political_change_needed'
            ]
            
            # Calculate mean scores by income quartile
            available_cols = [col for col in value_cols if col in self.data.columns]
            values_by_income = (
                self.data
                .groupby('income_quartile')[available_cols]
                .mean()
                .round(2)
            )
            
            # Create heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                values_by_income,
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Average Score'}
            )
            plt.title('Value Scores by Income Level')
            plt.xlabel('Values')
            plt.ylabel('Income Quartile')
            
            plt.tight_layout()
            save_plot(
                plt,
                self.figures_dir / 'income_value_relationships.png',
                show_plots
            )
            
            # Calculate correlations
            correlations = {}
            for col in available_cols:
                corr = self.data['family_monthly_income'].corr(self.data[col])
                correlations[col] = round(corr, 3)
            
            return {
                'by_income_level': values_by_income.to_dict(),
                'correlations': correlations
            }
            
        except Exception as e:
            print(f"Warning: Error in income analysis: {str(e)}")
            return {}

    def analyze_political_value_patterns(self, show_plots: bool = False):
        """Analyze patterns in political values and change views."""
        if 'political_change_needed' not in self.data.columns:
            return {}
            
        # Define relevant value columns
        value_cols = [
            'economic_freedom_score',
            'leadership_satisfaction_score',
            'tradition_reliance_score',
            'cultural_importance_score'
        ]
        
        results = {}
        
        # Calculate mean scores by political change view
        available_cols = [col for col in value_cols if col in self.data.columns]
        scores_by_view = (
            self.data
            .groupby('political_change_needed')[available_cols]
            .mean()
            .round(2)
        )
        
        results['mean_scores'] = scores_by_view.to_dict()
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        scores_by_view.plot(kind='bar')
        plt.title('Value Scores by Political Change View')
        plt.xlabel('Political Change View')
        plt.ylabel('Average Score')
        plt.legend(
            title='Values',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        plt.tight_layout()
        
        save_plot(
            plt,
            self.figures_dir / 'political_value_patterns.png',
            show_plots
        )
        
        # Analyze change views by program and income
        if 'income_quartile' in self.data.columns:
            change_by_income = pd.crosstab(
                self.data['income_quartile'],
                self.data['political_change_needed'],
                normalize='index'
            ) * 100
            
            results['by_income'] = change_by_income.to_dict()
        
        change_by_program = pd.crosstab(
            self.data['program'],
            self.data['political_change_needed'],
            normalize='index'
        ) * 100
        
        results['by_program'] = change_by_program.to_dict()
        
        return results

    def generate_summary_report(self, results: dict):
        """Generate summary report of value analysis."""
        report = [
            "Value Analysis Summary",
            "===================\n"
        ]
        
        # Add income-value relationships section
        if results.get('income_value_relationships'):
            report.extend([
                "Income-Value Relationships:",
                "------------------------"
            ])
            correlations = results['income_value_relationships']['correlations']
            for value, corr in correlations.items():
                report.append(f"{value}: {corr:.3f} correlation with income")
            
            report.append("\nValue Scores by Income Level:")
            for level, scores in results['income_value_relationships']['by_income_level'].items():
                report.append(f"\n{level}:")
                for value, score in scores.items():
                    report.append(f"  {value}: {score:.2f}")
        
        # Add political value patterns section
        if results.get('political_value_patterns'):
            report.extend([
                "\nPolitical Value Patterns:",
                "----------------------"
            ])
            if 'mean_scores' in results['political_value_patterns']:
                report.append("\nValue Scores by Political Change View:")
                for view, scores in results['political_value_patterns']['mean_scores'].items():
                    view_label = 'Change Needed' if view == 1 else 'No Change Needed'
                    report.append(f"\n{view_label}:")
                    for value, score in scores.items():
                        report.append(f"  {value}: {score:.2f}")
        
        # Value importance summary
        if results.get('value_importance'):
            report.extend([
                "\nValue Importance Scores:",
                "---------------------"
            ])
            sorted_values = sorted(
                results['value_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for col, score in sorted_values:
                value_name = self.value_columns.get(col, col)
                report.append(f"{value_name}: {score:.2f}")
        
        # Value correlations summary
        if results.get('value_correlations'):
            report.extend([
                "\nKey Value Correlations:",
                "--------------------",
                "See correlation heatmap for detailed relationships."
            ])
        
        # Add key relationships summary
        if results.get('key_relationships'):
            report.extend([
                "\nKey Relationship Analysis:",
                "------------------------"
            ])
            for title, stats in results['key_relationships'].items():
                report.extend([
                    f"\n{title}:",
                    f"Correlation: {stats['correlation']:.2f}",
                    f"Mean X: {stats['mean_x']:.2f}",
                    f"Mean Y: {stats['mean_y']:.2f}"
                ])
        
        # Save report
        report_path = self.reports_dir / 'value_summary.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def run_value_analysis(self, show_plots: bool = False) -> dict:
        """Run complete value analysis."""
        print("Running value analysis...")
        
        # Check required columns
        required_columns = set(self.value_columns.keys())
        missing_columns = required_columns - set(self.data.columns)
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            return {'error': f"Missing columns: {missing_columns}"}
        
        # Run analyses
        results = {
            'value_importance': self.analyze_value_importance(show_plots),
            'value_dimensions': self.analyze_value_dimensions(show_plots),
            'value_clusters': self.analyze_value_clusters(show_plots),
            'value_trends': self.analyze_value_trends(show_plots),
            'value_interactions': self.analyze_value_interactions(show_plots),
            'value_by_program': self.analyze_value_distribution_by_program(show_plots),
            'value_correlations': self.analyze_value_correlations(show_plots),
            'key_relationships': self.analyze_key_relationships(show_plots),
            'income_value_relationships': self.analyze_income_value_relationships(show_plots),
            'political_value_patterns': self.analyze_political_value_patterns(show_plots)
        }
        
        # Generate summary report
        summary = self.generate_summary_report(results)
        results['summary'] = summary
        
        print("Value analysis completed!")
        print(f"Figures saved in: {self.figures_dir}")
        print(f"Reports saved in: {self.reports_dir}")
        
        return results 