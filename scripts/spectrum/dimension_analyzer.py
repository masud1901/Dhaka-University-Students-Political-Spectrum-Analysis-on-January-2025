"""Analyzes political dimensions in survey data."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DimensionAnalyzer:
    def __init__(self, data: pd.DataFrame, indicators: dict):
        """Initialize with spectrum scores and indicator config."""
        self.data = data
        self.indicators = indicators
        
        # Add new indicators for income and political change
        self.socioeconomic_indicators = {
            'income_level': 'family_monthly_income',
            'political_change': 'political_change_needed'
        }
        
    def analyze_dimension_correlations(self):
        """Analyze correlations between different dimensions."""
        # Get available dimension columns
        dimension_cols = [f'{dim}_spectrum' for dim in self.indicators.keys() 
                        if f'{dim}_spectrum' in self.data.columns]
        
        if not dimension_cols:
            print("Warning: No dimension columns found for correlation analysis")
            return {}
            
        dimension_scores = self.data[dimension_cols]
        correlations = dimension_scores.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='RdBu', center=0)
        plt.title('Dimension Correlations')
        
        return correlations.to_dict()
        
    def perform_pca_analysis(self):
        """Perform PCA to identify main components of political ideology."""
        # Prepare data - get available features
        features = []
        for dim, config in self.indicators.items():
            available_features = [f for f in config['right_wing'] 
                                if f in self.data.columns]
            features.extend(available_features)
            
        if not features:
            print("Warning: No features available for PCA analysis")
            return {
                'components': [],
                'explained_variance': [],
                'feature_weights': {}
            }
            
        # Get data for available features
        X = self.data[features].copy()
        
        # Handle missing values if any
        X = X.fillna(X.mean())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA()
        components = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        
        return {
            'components': components,
            'explained_variance': explained_variance.tolist(),
            'feature_weights': pd.DataFrame(
                pca.components_,
                columns=features
            ).to_dict()
        }
        
    def analyze_socioeconomic_relationships(self):
        """Analyze relationships with socioeconomic factors."""
        relationships = {}
        
        # Analyze income relationships if available
        if 'family_monthly_income' in self.data.columns:
            income_correlations = {}
            for dim in self.indicators.keys():
                spectrum_col = f'{dim}_spectrum'
                if spectrum_col in self.data.columns:
                    corr = self.data['family_monthly_income'].corr(
                        self.data[spectrum_col]
                    )
                    income_correlations[dim] = round(corr, 3)
            relationships['income'] = income_correlations
        
        # Analyze political change relationships if available
        if 'political_change_needed' in self.data.columns:
            change_correlations = {}
            for dim in self.indicators.keys():
                spectrum_col = f'{dim}_spectrum'
                if spectrum_col in self.data.columns:
                    corr = self.data['political_change_needed'].corr(
                        self.data[spectrum_col]
                    )
                    change_correlations[dim] = round(corr, 3)
            relationships['political_change'] = change_correlations
            
        return relationships

    def run_dimension_analysis(self):
        """Run complete dimension analysis."""
        print("Running dimension analysis...")
        
        results = {
            'correlations': self.analyze_dimension_correlations(),
            'pca_results': self.perform_pca_analysis(),
            'socioeconomic': self.analyze_socioeconomic_relationships()
        }
        
        # Generate summary
        summary = [
            "Dimension Analysis Results:",
            "------------------------"
        ]
        
        # Add socioeconomic insights
        if results.get('socioeconomic'):
            if 'income' in results['socioeconomic']:
                summary.append("\nIncome Correlations:")
                for dim, corr in results['socioeconomic']['income'].items():
                    summary.append(f"{dim}: {corr:.2f}")
                    
            if 'political_change' in results['socioeconomic']:
                summary.append("\nPolitical Change Correlations:")
                for dim, corr in results['socioeconomic']['political_change'].items():
                    summary.append(f"{dim}: {corr:.2f}")
        
        results['summary'] = '\n'.join(summary)
        return results 