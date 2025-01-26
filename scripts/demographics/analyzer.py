"""Demographic analysis module for DU student survey."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from .utils import setup_plot_style, save_plot

class DemographicAnalyzer:
    """Analyzes demographic aspects of the survey data."""
    
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

    def analyze_program_distribution(self, show_plots: bool = False):
        """Analyze program type distribution."""
        prog_dist = self.data['program'].value_counts()
        
        if prog_dist.empty:
            print("Warning: No program distribution data available")
            return {}
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=prog_dist.index, y=prog_dist.values)
        plt.title('Distribution of Students by Program')
        plt.xlabel('Program Type')
        plt.ylabel('Number of Students')
        
        # Add percentage labels
        total = prog_dist.sum()
        for i, v in enumerate(prog_dist.values):
            percentage = (v / total) * 100
            ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Save plot
        save_plot(plt, self.figures_dir / 'program_distribution.png', show_plots)
        
        return prog_dist.to_dict()

    def analyze_session_distribution(self, show_plots: bool = False):
        """Analyze session-wise distribution."""
        session_dist = self.data['session'].value_counts().sort_index()
        
        if session_dist.empty:
            print("Warning: No session distribution data available")
            return {}
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=session_dist.index, y=session_dist.values)
        plt.title('Distribution of Students by Session')
        plt.xlabel('Session')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=45)
        
        # Add percentage labels
        total = session_dist.sum()
        for i, v in enumerate(session_dist.values):
            percentage = (v / total) * 100
            ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Save plot
        save_plot(plt, self.figures_dir / 'session_distribution.png', show_plots)
        
        return session_dist.to_dict()

    def analyze_program_session_relationship(self, show_plots: bool = False):
        """Analyze relationship between program and session."""
        if 'program' not in self.data.columns or 'session' not in self.data.columns:
            return {}
            
        # Create cross-tabulation
        cross_tab = pd.crosstab(
            self.data['program'],
            self.data['session'],
            normalize='columns'
        ) * 100
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            cross_tab,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.title('Program Distribution Across Sessions')
        plt.xlabel('Session')
        plt.ylabel('Program')
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'program_session_heatmap.png',
            show_plots
        )
        
        return cross_tab.to_dict()

    def analyze_reform_interests_by_program(self, show_plots: bool = False):
        """Analyze reform interests distribution by program."""
        # Only look for binary reform columns (created by value_transformer)
        reform_columns = [
            col for col in self.data.columns 
            if col.startswith('reform_') and col != 'reform_interests'
        ]
        
        if not reform_columns:
            print("Warning: No binary reform columns found")
            return {}
        
        try:
            # Calculate mean interest for each reform by program
            reform_by_program = (
                self.data
                .groupby('program')[reform_columns]
                .mean()
                .round(3)
            )
            
            # Create plot
            plt.figure(figsize=(12, 6))
            reform_by_program.T.plot(kind='bar')
            plt.title('Reform Interests by Program')
            plt.xlabel('Reform Type')
            plt.ylabel('Proportion of Students')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Program')
            
            # Save plot
            save_plot(
                plt,
                self.figures_dir / 'reform_interests_by_program.png',
                show_plots
            )
            
            return reform_by_program.to_dict()
            
        except Exception as e:
            print(f"Warning: Error in reform analysis: {str(e)}")
            print("Available columns:", self.data.columns.tolist())
            return {}

    def analyze_individualism_by_program(self, show_plots: bool = False):
        """Analyze individualism scores across different programs."""
        if 'individualism_score' not in self.data.columns:
            print("Warning: Individualism score not found")
            return {}

        # Calculate mean individualism score by program
        ind_by_program = self.data.groupby('program')['individualism_score'].agg([
            'mean', 'std', 'count'
        ]).round(2)

        # Create violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.data, x='program', y='individualism_score')
        plt.title('Distribution of Individualism Scores by Program')
        plt.xlabel('Program')
        plt.ylabel('Individualism Score')

        # Save plot
        save_plot(plt, self.figures_dir / 'individualism_by_program.png', show_plots)

        return ind_by_program.to_dict('index')

    def analyze_future_aspirations(self, show_plots: bool = False):
        """Analyze future aspirations distribution."""
        if 'future_aspiration' not in self.data.columns:
            print("Warning: Future aspiration data not found")
            return {}

        # Get aspiration distribution
        asp_dist = self.data['future_aspiration'].value_counts()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=asp_dist.index, y=asp_dist.values)
        plt.title('Distribution of Future Aspirations')
        plt.xlabel('Aspiration Type')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=45, ha='right')

        # Add percentage labels
        total = asp_dist.sum()
        for i, v in enumerate(asp_dist.values):
            percentage = (v / total) * 100
            ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')

        # Save plot
        save_plot(plt, self.figures_dir / 'future_aspirations.png', show_plots)

        return asp_dist.to_dict()

    def analyze_tradition_reliance(self, show_plots: bool = False):
        """Analyze tradition reliance patterns."""
        if 'tradition_reliance_score' not in self.data.columns:
            print("Warning: Tradition reliance score not found")
            return {}

        # Calculate mean tradition reliance by program and session
        trad_by_prog = self.data.groupby('program')['tradition_reliance_score'].mean()
        trad_by_session = self.data.groupby('session')['tradition_reliance_score'].mean()

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Program plot
        sns.barplot(x=trad_by_prog.index, y=trad_by_prog.values, ax=ax1)
        ax1.set_title('Tradition Reliance by Program')
        ax1.set_ylabel('Mean Tradition Reliance Score')

        # Session plot
        sns.barplot(x=trad_by_session.index, y=trad_by_session.values, ax=ax2)
        ax2.set_title('Tradition Reliance by Session')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        save_plot(plt, self.figures_dir / 'tradition_reliance.png', show_plots)

        return {
            'by_program': trad_by_prog.to_dict(),
            'by_session': trad_by_session.to_dict()
        }

    def analyze_value_correlations(self, show_plots: bool = False):
        """Analyze correlations between different value scores."""
        value_columns = [
            'individualism_score', 'tradition_reliance_score',
            'cultural_importance_score', 'leadership_satisfaction_score',
            'protest_support_score'
        ]
        
        # Filter only available columns
        available_cols = [col for col in value_columns if col in self.data.columns]
        
        if len(available_cols) < 2:
            print("Warning: Not enough value columns for correlation analysis")
            return {}

        # Calculate correlation matrix
        corr_matrix = self.data[available_cols].corr().round(2)

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdBu',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Correlations Between Value Scores')
        
        # Save plot
        save_plot(plt, self.figures_dir / 'value_correlations.png', show_plots)

        return corr_matrix.to_dict()

    def analyze_income_distribution(self, show_plots: bool = False):
        """Analyze family monthly income distribution."""
        if 'family_monthly_income' not in self.data.columns:
            print("Warning: Income data not found")
            return {}
            
        # Calculate basic statistics
        income_stats = self.data['family_monthly_income'].describe()
        
        # Create income brackets for visualization
        brackets = [0, 20000, 40000, 60000, 100000, float('inf')]
        labels = [
            '0-20k', '20k-40k', '40k-60k', 
            '60k-100k', '100k+'
        ]
        
        self.data['income_bracket'] = pd.cut(
            self.data['family_monthly_income'],
            bins=brackets,
            labels=labels,
            right=False
        )
        
        # Calculate distribution
        income_dist = self.data['income_bracket'].value_counts().sort_index()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=income_dist.index, y=income_dist.values)
        plt.title('Family Monthly Income Distribution')
        plt.xlabel('Income Bracket (BDT)')
        plt.ylabel('Number of Students')
        plt.xticks(rotation=45)
        
        # Add percentage labels
        total = income_dist.sum()
        for i, v in enumerate(income_dist.values):
            percentage = (v / total) * 100
            ax.text(i, v, f'{percentage:.1f}%', ha='center', va='bottom')
        
        # Save plot
        save_plot(
            plt, 
            self.figures_dir / 'income_distribution.png',
            show_plots
        )
        
        return {
            'stats': income_stats.to_dict(),
            'distribution': income_dist.to_dict()
        }

    def analyze_income_program_relationship(self, show_plots: bool = False):
        """Analyze relationship between income and program."""
        if 'family_monthly_income' not in self.data.columns:
            return {}
            
        # Calculate mean income by program
        income_by_program = (
            self.data
            .groupby('program')['family_monthly_income']
            .agg(['mean', 'median', 'std'])
            .round(2)
        )
        
        # Create box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=self.data,
            x='program',
            y='family_monthly_income'
        )
        plt.title('Income Distribution by Program')
        plt.xlabel('Program')
        plt.ylabel('Monthly Family Income (BDT)')
        
        # Save plot
        save_plot(
            plt,
            self.figures_dir / 'income_by_program.png',
            show_plots
        )
        
        return income_by_program.to_dict('index')

    def generate_summary_report(self, results: dict):
        """Generate summary report of demographic analysis."""
        report = [
            "Demographic Analysis Summary",
            "=========================\n"
        ]
        
        # Program Distribution
        if results.get('program_distribution'):
            report.extend([
                "Program Distribution:",
                "-----------------"
            ])
            total_students = sum(results['program_distribution'].values())
            for prog, count in results['program_distribution'].items():
                percentage = (count / total_students) * 100
                report.append(
                    f"{prog}: {count} students ({percentage:.1f}%)"
                )
        else:
            report.append("No program distribution data available")
        
        report.append("")  # Empty line for spacing
        
        # Session Distribution
        if results.get('session_distribution'):
            report.extend([
                "Session Distribution:",
                "------------------"
            ])
            total_students = sum(results['session_distribution'].values())
            for session, count in results['session_distribution'].items():
                percentage = (count / total_students) * 100
                report.append(
                    f"{session}: {count} students ({percentage:.1f}%)"
                )
        else:
            report.append("No session distribution data available")
        
        # Program-Session Relationship
        if results.get('program_session_relationship'):
            report.extend([
                "\nProgram-Session Distribution:",
                "--------------------------",
                "See heatmap visualization for detailed breakdown."
            ])
        
        # Reform Interests
        if results.get('reform_interests_by_program'):
            report.extend([
                "\nReform Interests by Program:",
                "-------------------------",
                "See bar plot visualization for detailed breakdown."
            ])
        
        # Individualism Analysis
        if results.get('individualism_analysis'):
            report.extend([
                "\nIndividualism Analysis:",
                "------------------"
            ])
            for prog, stats in results['individualism_analysis'].items():
                report.append(f"{prog}:")
                for stat, value in stats.items():
                    report.append(f"  {stat}: {value}")
        
        # Future Aspirations
        if results.get('future_aspirations'):
            report.extend([
                "\nFuture Aspirations:",
                "------------------"
            ])
            total_aspirations = sum(results['future_aspirations'].values())
            for aspiration, count in results['future_aspirations'].items():
                percentage = (count / total_aspirations) * 100
                report.append(f"{aspiration}: {count} students ({percentage:.1f}%)")
        
        # Tradition Reliance
        if results.get('tradition_reliance'):
            report.extend([
                "\nTradition Reliance:",
                "------------------",
                "\nBy Program:"
            ])
            if 'by_program' in results['tradition_reliance']:
                for prog, score in results['tradition_reliance']['by_program'].items():
                    report.append(f"  {prog}: {score:.2f}")
            
            report.append("\nBy Session:")
            if 'by_session' in results['tradition_reliance']:
                for session, score in results['tradition_reliance']['by_session'].items():
                    report.append(f"  {session}: {score:.2f}")
        
        # Value Correlations
        if results.get('value_correlations'):
            report.extend([
                "\nValue Correlations:",
                "------------------"
            ])
            # Correctly handle the correlation matrix dictionary
            for var1 in results['value_correlations']:
                for var2, corr in results['value_correlations'][var1].items():
                    if var1 < var2:  # Only show each correlation once
                        report.append(f"{var1} - {var2}: {corr:.2f}")
        
        # Income Distribution
        if results.get('income_distribution'):
            report.extend([
                "\nIncome Distribution:",
                "------------------"
            ])
            for stat, value in results['income_distribution']['stats'].items():
                report.append(f"{stat}: {value}")
        
        # Income-Program Relationship
        if results.get('income_program'):
            report.extend([
                "\nIncome Distribution by Program:",
                "------------------"
            ])
            for prog, stats in results['income_program'].items():
                report.append(f"{prog}:")
                for stat, value in stats.items():
                    report.append(f"  {stat}: {value}")
        
        # Save report
        report_path = self.reports_dir / 'demographic_summary.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

    def run_demographic_analysis(self, show_plots: bool = False) -> dict:
        """Run complete demographic analysis."""
        print("Running demographic analysis...")
        
        # Check if required columns exist
        required_columns = {'program', 'session'}
        missing_columns = required_columns - set(self.data.columns)
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            print("Available columns:", self.data.columns.tolist())
            return {'error': f"Missing columns: {missing_columns}"}
        
        # Run analyses
        results = {
            'program_distribution': self.analyze_program_distribution(show_plots),
            'session_distribution': self.analyze_session_distribution(show_plots),
            'program_session_relationship': self.analyze_program_session_relationship(show_plots),
            'reform_interests_by_program': self.analyze_reform_interests_by_program(show_plots),
            'individualism_analysis': self.analyze_individualism_by_program(show_plots),
            'future_aspirations': self.analyze_future_aspirations(show_plots),
            'tradition_reliance': self.analyze_tradition_reliance(show_plots),
            'value_correlations': self.analyze_value_correlations(show_plots),
            'income_distribution': self.analyze_income_distribution(show_plots),
            'income_program': self.analyze_income_program_relationship(show_plots),
        }
        
        # Generate summary report
        summary = self.generate_summary_report(results)
        results['summary'] = summary
        
        print("Demographic analysis completed!")
        print(f"Figures saved in: {self.figures_dir}")
        print(f"Reports saved in: {self.reports_dir}")
        
        return results 