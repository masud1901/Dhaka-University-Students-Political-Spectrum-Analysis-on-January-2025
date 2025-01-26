"""Advanced clustering analysis for political spectrum."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from .utils import setup_plot_style, save_plot, scale_features
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class ClusterAnalyzer:
    """Analyzes political clusters using multiple algorithms."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with spectrum scores."""
        self.data = data
        self.output_dir = Path(__file__).parent
        self.figures_dir = self.output_dir / 'figures' / 'clusters'
        self.reports_dir = self.output_dir / 'reports'
        
        # Create output directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting style
        setup_plot_style()
        
        # Setup imputer
        self.imputer = SimpleImputer(strategy='mean')

    def _preprocess_data(self, data):
        """Preprocess data by handling NaN values and scaling."""
        # First impute missing values
        imputed_data = self.imputer.fit_transform(data)
        
        # Then scale the features
        scaled_data = scale_features(imputed_data)
        
        return scaled_data

    def perform_hierarchical_clustering(self, X, max_clusters=10):
        """Perform hierarchical clustering analysis."""
        # Calculate linkage matrix
        linkage_matrix = linkage(X, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=max_clusters,
            show_leaf_counts=True
        )
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Cluster')
        plt.ylabel('Distance')
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'hierarchical_dendrogram.png',
            show_plots=False
        )
        
        return linkage_matrix

    def evaluate_cluster_stability(self, X, k_range=range(2, 11)):
        """Evaluate cluster stability using multiple metrics."""
        stability_metrics = {
            'silhouette': [],
            'calinski_harabasz': [],
            'inertia': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Calculate stability metrics
            stability_metrics['silhouette'].append(silhouette_score(X, labels))
            stability_metrics['calinski_harabasz'].append(
                calinski_harabasz_score(X, labels)
            )
            stability_metrics['inertia'].append(kmeans.inertia_)
        
        # Plot stability metrics
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Silhouette score
        axes[0].plot(k_range, stability_metrics['silhouette'], 'bo-')
        axes[0].set_title('Silhouette Score')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Score')
        
        # Calinski-Harabasz score
        axes[1].plot(k_range, stability_metrics['calinski_harabasz'], 'ro-')
        axes[1].set_title('Calinski-Harabasz Score')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Score')
        
        # Inertia (within-cluster sum of squares)
        axes[2].plot(k_range, stability_metrics['inertia'], 'go-')
        axes[2].set_title('Inertia')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('Score')
        
        plt.tight_layout()
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'cluster_stability.png',
            show_plots=False
        )
        
        return stability_metrics

    def analyze_cluster_profiles(self, X, labels):
        """Analyze detailed profiles of each cluster."""
        profiles = {}
        df = pd.DataFrame(X, columns=self.data.columns)
        df['Cluster'] = labels
        
        for cluster in np.unique(labels):
            cluster_data = df[df['Cluster'] == cluster]
            
            # Calculate basic statistics
            stats = cluster_data.describe()
            
            # Calculate feature importance
            cluster_mean = cluster_data.mean()
            overall_mean = df.mean()
            feature_importance = (cluster_mean - overall_mean).abs()
            
            # Store results
            profiles[cluster] = {
                'size': len(cluster_data),
                'statistics': stats.to_dict(),
                'distinctive_features': feature_importance.nlargest(3).to_dict(),
                'mean_profile': cluster_mean.to_dict()
            }
        
        return profiles

    def find_optimal_clusters(self, max_clusters=8):
        """Find optimal number of clusters using silhouette score."""
        X = StandardScaler().fit_transform(self.data)
        
        silhouette_scores = []
        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        return {
            'optimal_clusters': optimal_clusters,
            'silhouette_scores': silhouette_scores
        }

    def compare_clustering_methods(self):
        """Compare K-means and hierarchical clustering."""
        X = StandardScaler().fit_transform(self.data)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=4)
        hierarchical_labels = hierarchical.fit_predict(X)
        
        # Compare cluster assignments
        comparison = pd.DataFrame({
            'kmeans': kmeans_labels,
            'hierarchical': hierarchical_labels
        })
        
        # Visualize cluster comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.data, x='economic_spectrum', y='social_spectrum', 
                       hue=kmeans_labels, palette='deep')
        plt.title('K-means Clusters')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=self.data, x='economic_spectrum', y='social_spectrum', 
                       hue=hierarchical_labels, palette='deep')
        plt.title('Hierarchical Clusters')
        
        return {
            'kmeans_labels': kmeans_labels.tolist(),
            'hierarchical_labels': hierarchical_labels.tolist(),
            'cluster_sizes': {
                'kmeans': pd.Series(kmeans_labels).value_counts().to_dict(),
                'hierarchical': pd.Series(hierarchical_labels).value_counts().to_dict()
            }
        }

    def visualize_clusters(self, X, labels, method_name):
        """Create detailed visualization of clusters."""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with first two dimensions
        scatter = plt.scatter(
            X[:, 0], X[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6
        )
        
        # Add cluster centers if available
        unique_labels = np.unique(labels)
        centers = np.array([
            X[labels == label].mean(axis=0)
            for label in unique_labels
        ])
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster Centers'
        )
        
        plt.title(f'Cluster Analysis using {method_name}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / f'clusters_{method_name.lower()}.png',
            show_plots=False
        )

    def analyze_cluster_characteristics(self, labels, method_name: str):
        """Analyze characteristics of each cluster."""
        analysis_df = self.data.copy()
        analysis_df['Cluster'] = labels
        
        characteristics = {}
        
        # Analyze each cluster
        for cluster in np.unique(labels):
            cluster_data = analysis_df[analysis_df['Cluster'] == cluster]
            
            # Calculate basic statistics
            stats = cluster_data.describe()
            
            # Calculate mean income if available
            if 'family_monthly_income' in cluster_data.columns:
                mean_income = cluster_data['family_monthly_income'].mean()
            else:
                mean_income = None
                
            # Calculate political change preference if available
            if 'political_change_needed' in cluster_data.columns:
                change_pref = (
                    cluster_data['political_change_needed']
                    .value_counts(normalize=True)
                    .to_dict()
                )
            else:
                change_pref = None
            
            characteristics[cluster] = {
                'size': len(cluster_data),
                'stats': stats.to_dict(),
                'mean_income': mean_income,
                'political_change_preference': change_pref,
                'program_distribution': (
                    cluster_data['program']
                    .value_counts(normalize=True)
                    .to_dict()
                )
            }
        
        return characteristics

    def generate_cluster_report(self, results: dict):
        """Generate detailed report of clustering analysis."""
        report = [
            "Political Clustering Analysis Report",
            "================================\n"
        ]
        
        # Optimal cluster analysis
        if results.get('optimal_clusters'):
            report.extend([
                "Optimal Number of Clusters:",
                "------------------------",
                f"Silhouette optimal: {results['optimal_clusters']['optimal_clusters']}",
                f"Silhouette scores: {', '.join(map(str, results['optimal_clusters']['silhouette_scores']))}\n"
            ])
        
        # Clustering method comparison
        if results.get('clustering_comparison'):
            report.extend([
                "Clustering Method Comparison:",
                "--------------------------"
            ])
            
            for method, data in results['clustering_comparison'].items():
                report.extend([
                    f"\n{method} Clustering:",
                    f"Number of clusters: {len(np.unique(data['labels']))}",
                    "Cluster sizes: " + ", ".join(
                        f"{i}: {sum(data['labels'] == i)}"
                        for i in np.unique(data['labels'])
                    )
                ])
        
        # Save report
        report_path = self.reports_dir / 'clustering_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report) 