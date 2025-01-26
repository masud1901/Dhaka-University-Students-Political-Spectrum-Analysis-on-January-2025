"""Main data cleaning module that orchestrates the cleaning process."""
import pandas as pd
from pathlib import Path
from .column_normalizer import ColumnNormalizer
from .value_transformer import ValueTransformer


class StudentSurveyDataCleaner:
    """Class to handle DU student survey data cleaning pipeline."""
    
    def __init__(self):
        """Initialize the data cleaner with normalizer and transformer."""
        self.normalizer = ColumnNormalizer()
        self.transformer = ValueTransformer()

    def _save_dataframe(self, df: pd.DataFrame, filename: str):
        """Save dataframe to CSV file."""
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")

    def run_data_cleaning(
        self,
        df: pd.DataFrame,
        save_intermediate: bool = True,
        save_final: bool = True
    ) -> pd.DataFrame:
        """Main entry point for data cleaning pipeline."""
        print("Starting data cleaning process...")
        
        # Clean the data
        df_clean = df.copy()
        
        # Step 1: Normalize column names
        print("Normalizing column names...")
        df_normalized = self.normalizer.normalize_columns(df_clean)
        
        # Save normalized data if requested
        if save_intermediate:
            self._save_dataframe(
                df_normalized,
                'normalized_student_survey.csv'
            )
        
        # Step 2: Transform values
        print("Transforming values...")
        df_transformed = self.transformer.transform_values(df_normalized)
        
        # Drop timestamp if exists
        if 'timestamp' in df_transformed.columns:
            df_transformed = df_transformed.drop('timestamp', axis=1)
        
        # Save final cleaned data if requested
        if save_final:
            self._save_dataframe(
                df_transformed,
                'cleaned_student_survey.csv'
            )
        
        print("Data cleaning completed successfully!")
        return df_transformed