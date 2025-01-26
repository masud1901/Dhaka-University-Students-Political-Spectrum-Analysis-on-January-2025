"""Analysis module for student perspectives and views."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from .utils import setup_plot_style, save_plot, calculate_percentage


class PerspectiveAnalyzer:
    """Analyzes student perspectives on various social and economic issues."""
    
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
        
        # Define perspective categories with metadata
        self.perspective_categories = {
            'economic': {
                'columns': [
                    'economic_freedom_score',
                    'market_over_regulation',
                    'economic_growth_over_environment'
                ],
                'label': 'Economic Views',
                'description': 'Views on economic policy and regulation'
            },
            'social': {
                'columns': [
                    'individual_freedom_over_harmony',
                    'protest_support_score',
                    'stability_over_justice'
                ],
                'label': 'Social Views',
                'description': 'Views on social issues and justice'
            },
            'cultural': {
                'columns': [
                    'cultural_importance_score',
                    'tradition_reliance_score',
                    'traditional_over_progressive'
                ],
                'label': 'Cultural Views',
                'description': 'Views on cultural traditions and change'
            },
            'governance': {
                'columns': [
                    'leadership_satisfaction_score',
                    'central_over_local'
                ],
                'label': 'Governance Views',
                'description': 'Views on leadership and governance'
            }
        }

    def analyze_economic_views(self):
        """Analyze economic policy preferences."""
        economic_stats = {
            'mean_freedom_score': self.data['economic_freedom_score'].mean(),
            'median_freedom_score': self.data['economic_freedom_score'].median(),
            'pro_market_percentage': calculate_percentage(
                self.data,
                'market_over_regulation',
                lambda x: x == 1
            ),
            'growth_over_environment': calculate_percentage(
                self.data,
                'economic_growth_over_environment',
                lambda x: x == 1
            )
        }
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Economic freedom score distribution
        sns.histplot(
            data=self.data,
            x='economic_freedom_score',
            bins=20,
            kde=True
        )
        plt.title('Distribution of Economic Freedom Scores')
        plt.xlabel('Economic Freedom Score')
        plt.ylabel('Count')
        
        # Add mean and median lines
        plt.axvline(
            economic_stats['mean_freedom_score'],
            color='r',
            linestyle='--',
            label=f"Mean: {economic_stats['mean_freedom_score']:.2f}"
        )
        plt.axvline(
            economic_stats['median_freedom_score'],
            color='g',
            linestyle='--',
            label=f"Median: {economic_stats['median_freedom_score']:.2f}"
        )
        plt.legend()
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'economic_views.png',
            show_plots=False
        )
        
        return economic_stats

    def analyze_social_views(self):
        """Analyze social and cultural perspectives."""
        social_stats = {
            'cultural_importance': self.data['cultural_importance_score'].mean(),
            'tradition_reliance': self.data['tradition_reliance_score'].mean(),
            'protest_support': self.data['protest_support_score'].mean(),
            'prefer_individual_freedom': calculate_percentage(
                self.data,
                'individual_freedom_over_harmony',
                lambda x: x == 1
            ),
            'prefer_stability': calculate_percentage(
                self.data,
                'stability_over_justice',
                lambda x: x == 1
            )
        }
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cultural importance
        sns.boxplot(
            data=self.data,
            y='cultural_importance_score',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Cultural Importance Scores')
        
        # Tradition reliance
        sns.boxplot(
            data=self.data,
            y='tradition_reliance_score',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Tradition Reliance Scores')
        
        # Protest support
        sns.boxplot(
            data=self.data,
            y='protest_support_score',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Protest Support Scores')
        
        # Binary preferences
        binary_data = pd.DataFrame({
            'Preference': ['Individual Freedom', 'Social Harmony', 
                         'Stability', 'Justice'],
            'Percentage': [
                social_stats['prefer_individual_freedom'],
                100 - social_stats['prefer_individual_freedom'],
                social_stats['prefer_stability'],
                100 - social_stats['prefer_stability']
            ],
            'Category': ['Freedom-Harmony', 'Freedom-Harmony',
                        'Stability-Justice', 'Stability-Justice']
        })
        
        sns.barplot(
            data=binary_data,
            x='Category',
            y='Percentage',
            hue='Preference',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Social Preferences')
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'social_views.png',
            show_plots=False
        )
        
        return social_stats

    def analyze_governance_views(self):
        """Analyze views on governance and leadership."""
        governance_stats = {
            'leadership_satisfaction': (
                self.data['leadership_satisfaction_score'].mean()
            ),
            'leadership_approval': calculate_percentage(
                self.data,
                'leadership_satisfaction_score',
                lambda x: x >= 4
            ),
            'prefer_central': calculate_percentage(
                self.data,
                'central_over_local',
                lambda x: x == 1
            )
        }
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Leadership satisfaction distribution
        sns.histplot(
            data=self.data,
            x='leadership_satisfaction_score',
            bins=20,
            kde=True
        )
        plt.title('Distribution of Leadership Satisfaction')
        plt.xlabel('Satisfaction Score')
        plt.ylabel('Count')
        
        # Add mean line
        plt.axvline(
            governance_stats['leadership_satisfaction'],
            color='r',
            linestyle='--',
            label=f"Mean: {governance_stats['leadership_satisfaction']:.2f}"
        )
        plt.legend()
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'governance_views.png',
            show_plots=False
        )
        
        return governance_stats

    def analyze_perspective_profiles(self, show_plots: bool = False):
        """Analyze distinct perspective profiles using clustering."""
        # Get all perspective columns
        perspective_cols = []
        for category in self.perspective_categories.values():
            perspective_cols.extend(
                col for col in category['columns'] 
                if col in self.data.columns
            )
            
        if not perspective_cols:
            print("Warning: No perspective columns available for clustering")
            return {}
            
        # Prepare data for clustering
        X = self.data[perspective_cols].copy()
        
        # Convert Int64 columns to float64 before handling missing values
        for col in X.columns:
            if X[col].dtype == 'Int64':
                X[col] = X[col].astype('float64')
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters
        silhouette_scores = []
        K = range(2, 6)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            
        optimal_k = K[np.argmax(silhouette_scores)]
        
        # Perform clustering with optimal K
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.data['perspective_cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze cluster characteristics
        cluster_profiles = pd.DataFrame()
        for col in perspective_cols:
            cluster_profiles[col] = [
                self.data[self.data['perspective_cluster'] == i][col].mean()
                for i in range(optimal_k)
            ]
            
        # Create visualization
        if show_plots:
            plt.figure(figsize=(12, 6))
            
            # Plot silhouette scores
            plt.subplot(1, 2, 1)
            plt.plot(K, silhouette_scores, 'bo-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Silhouette Score')
            plt.title('Optimal Number of Clusters')
            
            # Plot cluster characteristics
            plt.subplot(1, 2, 2)
            sns.heatmap(
                cluster_profiles,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu'
            )
            plt.title('Cluster Characteristics')
            plt.xlabel('Perspective Dimensions')
            plt.ylabel('Cluster')
            
            plt.tight_layout()
            save_plot(
                plt,
                self.figures_dir / 'perspective_clusters.png',
                show_plots
            )
            
        return {
            'optimal_clusters': optimal_k,
            'silhouette_scores': silhouette_scores,
            'cluster_profiles': cluster_profiles.to_dict()
        }

    def analyze_perspective_trends(self, show_plots: bool = False):
        """Analyze trends in perspectives across sessions."""
        trends = {}
        
        for category, info in self.perspective_categories.items():
            category_trends = {}
            for col in info['columns']:
                if col in self.data.columns:
                    # Calculate mean scores by session
                    session_means = (
                        self.data
                        .groupby('session')[col]
                        .agg(['mean', 'std', 'count'])
                        .round(3)
                    )
                    category_trends[col] = session_means.to_dict('index')
                    
                    # Perform one-way ANOVA
                    sessions = self.data['session'].unique()
                    session_groups = [
                        self.data[self.data['session'] == session][col]
                        for session in sessions
                    ]
                    f_stat, p_val = stats.f_oneway(*session_groups)
                    
                    category_trends[f'{col}_anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_val
                    }
            
            trends[category] = category_trends
        
        # Visualize trends
        plt.figure(figsize=(15, 10))
        for i, (category, info) in enumerate(self.perspective_categories.items()):
            plt.subplot(2, 2, i+1)
            for col in info['columns']:
                if col in self.data.columns:
                    means = self.data.groupby('session')[col].mean()
                    plt.plot(means.index, means.values, marker='o', label=col)
            plt.title(f'{info["label"]} Trends')
            plt.xlabel('Session')
            plt.ylabel('Mean Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        save_plot(plt, self.figures_dir / 'perspective_trends.png', show_plots)
        
        return trends

    def analyze_perspective_interactions(self, show_plots: bool = False):
        """Analyze interactions between different perspective dimensions."""
        interactions = []
        
        # Get all perspective columns
        perspective_cols = []
        for category in self.perspective_categories.values():
            perspective_cols.extend(category['columns'])
        
        # Calculate correlations and chi-square tests
        for i, col1 in enumerate(perspective_cols):
            for col2 in perspective_cols[i+1:]:
                if col1 in self.data.columns and col2 in self.data.columns:
                    # Calculate correlation
                    corr = self.data[col1].corr(self.data[col2])
                    
                    # Perform chi-square test
                    contingency = pd.crosstab(
                        self.data[col1].round(),
                        self.data[col2].round()
                    )
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                    
                    interactions.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr,
                        'chi2_statistic': chi2,
                        'p_value': p_val
                    })
        
        # Create network visualization
        if show_plots and interactions:
            import networkx as nx
            G = nx.Graph()
            
            # Add nodes
            for col in perspective_cols:
                G.add_node(col)
            
            # Add edges for significant correlations
            for interaction in interactions:
                if abs(interaction['correlation']) > 0.3:
                    G.add_edge(
                        interaction['variable1'],
                        interaction['variable2'],
                        weight=abs(interaction['correlation'])
                    )
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(
                G, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=2000,
                font_size=8,
                font_weight='bold'
            )
            
            save_plot(plt, self.figures_dir / 'perspective_network.png', show_plots)
        
        return interactions

    def generate_perspective_report(self, results: dict):
        """Generate detailed report of perspective analysis."""
        report = [
            "Student Perspective Analysis Report",
            "================================\n"
        ]
        
        # Economic views
        if results.get('economic_views'):
            report.extend([
                "Economic Perspectives:",
                "-------------------",
                f"Mean Economic Freedom Score: {results['economic_views']['mean_freedom_score']:.2f}",
                f"Pro-Market Percentage: {results['economic_views']['pro_market_percentage']:.1f}%",
                f"Growth over Environment: {results['economic_views']['growth_over_environment']:.1f}%\n"
            ])
        
        # Social views
        if results.get('social_views'):
            report.extend([
                "Social Perspectives:",
                "------------------",
                f"Cultural Importance: {results['social_views']['cultural_importance']:.2f}",
                f"Tradition Reliance: {results['social_views']['tradition_reliance']:.2f}",
                f"Individual Freedom Preference: {results['social_views']['prefer_individual_freedom']:.1f}%\n"
            ])
        
        # Governance views
        if results.get('governance_views'):
            report.extend([
                "Governance Perspectives:",
                "---------------------",
                f"Leadership Satisfaction: {results['governance_views']['leadership_satisfaction']:.2f}",
                f"Leadership Approval Rate: {results['governance_views']['leadership_approval']:.1f}%",
                f"Centralization Preference: {results['governance_views']['prefer_central']:.1f}%"
            ])
        
        # Save report
        report_path = self.reports_dir / 'perspective_analysis.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def run_perspective_analysis(self, show_plots: bool = False) -> dict:
        """Run complete perspective analysis."""
        print("Running perspective analysis...")
        
        results = {
            'economic_views': self.analyze_economic_views(),
            'social_views': self.analyze_social_views(),
            'governance_views': self.analyze_governance_views(),
            'perspective_profiles': self.analyze_perspective_profiles(show_plots),
            'perspective_trends': self.analyze_perspective_trends(show_plots),
            'perspective_interactions': self.analyze_perspective_interactions(show_plots)
        }
        
        # Generate summary report
        summary = self.generate_perspective_report(results)
        results['summary'] = summary
        
        print("Perspective analysis completed!")
        print(f"Figures saved in: {self.figures_dir}")
        print(f"Reports saved in: {self.reports_dir}")
        
        return results 