"""Module for transforming survey values into numerical format."""
import pandas as pd
import re


class ValueTransformer:
    """Handles value transformation for survey data."""
    
    def __init__(self):
        """Initialize value mapping dictionaries."""
        # Binary (Yes/No) mappings
        self.binary_map = {
            'yes': 1,
            'no': 0,
            'maybe': None  # Handle uncertain responses if any
        }
        
        # 5-point Likert scale mappings
        self.agreement_map = {
            'strongly agree': 5,
            'agree': 4,
            'neutral': 3,
            'disagree': 2,
            'strongly disagree': 1
        }
        
        # Satisfaction scale
        self.satisfaction_map = {
            'very satisfied': 5,
            'satisfied': 4,
            'neutral': 3,
            'dissatisfied': 2,
            'very dissatisfied': 1,
            '5': 5, '4': 4, '3': 3, '2': 2, '1': 1
        }
        
        # Frequency scale
        self.frequency_map = {
            'always': 5,
            'often': 4,
            'sometimes': 3,
            'rarely': 2,
            'never': 1
        }
        
        # Program type mapping
        self.program_map = {
            'honours': 0,
            'masters': 1
        }
        
        # Keep these as categorical (no transformation needed)
        self.categorical_columns = {
            'name',
            'session',
            'preferred_reform_areas',  # Multiple choice
            'self_in_10_years'      # Open-ended
        }
        
        # Columns to drop
        self.columns_to_drop = {
            'traditional_food_representation'  # Dropping food column as requested
        }

        # Add individualism scoring for future aspirations
        self.individualism_map = {
            'want to explore business': 5,      # Highest individualism - entrepreneurial
            'study abroad': 4,                  # High individualism - personal growth
            'career in my country': 3,          # Moderate - could be either
            'government serviceholder': 2,      # More collectivist - institutional
            'i just want to be happy with my family': 1  # Most collectivist - family-oriented
        }

        # Columns to transform
        self.columns_to_transform = [
            # existing columns...
            'family_monthly_income',
            'political_change_needed'
        ]

    def _format_phone_number(self, phone: str) -> str:
        """Format phone numbers to standard Bangladesh format."""
        if pd.isna(phone):
            return None
            
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', str(phone))
        
        # Handle different cases
        if len(digits) == 11:  # If only the number without country code
            formatted = f'+88{digits}'
        elif len(digits) == 13 and digits.startswith('88'):
            formatted = f'+{digits}'
        elif len(digits) == 14 and digits.startswith('088'):
            formatted = f'+{digits[1:]}'
        else:
            # If invalid length, return original with note
            return f"INVALID:{phone}"
        
        return formatted

    def _transform_phone_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform phone numbers to standard format."""
        if 'phone' in df.columns:
            df['phone'] = df['phone'].apply(self._format_phone_number)
        return df

    def _clean_text_response(self, text: str) -> str:
        """Clean text responses for consistent mapping."""
        if pd.isna(text):
            return text
        return (
            str(text)
            .lower()
            .strip()
            .replace('"', '')
            .replace('.', '')
            .replace('  ', ' ')
        )

    def _detect_and_map_scale(self, series: pd.Series) -> tuple[dict, str]:
        """Detect the type of scale and return appropriate mapping."""
        # Get unique values, cleaned and lowercase
        unique_vals = set(
            str(val).lower().strip().replace('"', '').replace('.', '')
            for val in series.unique() if pd.notna(val)
        )
        
        # Check if values are already numeric and within 1-5
        numeric_vals = {'1', '2', '3', '4', '5'}
        if unique_vals.issubset(numeric_vals):
            return {str(i): i for i in range(1, 6)}, 'numeric'
        
        # Check for scaled numeric values (e.g., 10, 20, ..., 50)
        scaled_numeric_vals = {'10', '20', '30', '40', '50'}
        if unique_vals.issubset(scaled_numeric_vals):
            return {str(i): i//10 for i in range(10, 60, 10)}, 'scaled_numeric'
            
        # Check for agreement patterns
        agreement_patterns = {'strongly agree', 'agree', 'neutral', 'disagree', 'strongly disagree'}
        if any(val in unique_vals for val in agreement_patterns):
            return self.agreement_map, 'agreement'
            
        # Check for satisfaction patterns
        satisfaction_patterns = {'very satisfied', 'satisfied', 'dissatisfied', 'very dissatisfied'}
        if any(val in unique_vals for val in satisfaction_patterns):
            return self.satisfaction_map, 'satisfaction'
            
        # Check for frequency patterns
        frequency_patterns = {'always', 'often', 'sometimes', 'rarely', 'never'}
        if any(val in unique_vals for val in frequency_patterns):
            return self.frequency_map, 'frequency'
            
        print(f"Warning: Unknown scale type. Unique values: {unique_vals}")
        return None, 'unknown'

    def _transform_scale_column(self, df: pd.DataFrame, col: str, score_col: str) -> pd.DataFrame:
        """Transform a column to a 1-5 scale based on its content."""
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataframe")
            return df
            
        print(f"Processing scale for {col}")
        # Clean the text first
        df[col] = df[col].apply(self._clean_text_response)
        
        # Detect scale type and get mapping
        mapping, scale_type = self._detect_and_map_scale(df[col])
        
        if mapping is None:
            print(f"Could not determine mapping for {col}")
            return df
            
        print(f"Detected scale type: {scale_type}")
        
        if scale_type == 'scaled_numeric':
            # Convert to float to handle NaN, perform floor division, and convert to nullable integer
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[score_col] = (df[col] // 10).astype('Int64')
        else:
            # Apply existing mapping
            df[score_col] = df[col].map(mapping)
        
        # Check for NaN values
        if pd.isna(df[score_col]).any():
            print(f"Warning: NaN values in {score_col}")
            print(f"Unique values in {col} after cleaning:", df[col].unique())
            
        # Drop original column
        df = df.drop(col, axis=1)
        return df

    def _transform_binary_choices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform binary choice responses to 1/0."""
        binary_columns = [
            'economic_growth_over_environment',
            'individual_freedom_over_harmony',
            'traditional_over_progressive',
            'central_over_local',
            'market_over_regulation',
            'stability_over_justice'
        ]
        
        for col in binary_columns:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .apply(self._clean_text_response)
                    .map(self.binary_map)
                )
        return df

    def _transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables where appropriate."""
        if 'program' in df.columns:
            df['program'] = df['program'].apply(self._clean_text_response)
            df['program_type'] = df['program'].map(self.program_map)
            
        # Transform self_in_10_years to individualism score
        if 'self_in_10_years' in df.columns:
            # Clean and standardize the responses
            df['self_in_10_years'] = df['self_in_10_years'].apply(self._clean_text_response)
            
            # Create individualism score (1-5 scale)
            df['individualism_score'] = df['self_in_10_years'].map(self.individualism_map)
            
            # Keep original response for qualitative analysis
            df['future_aspiration'] = df['self_in_10_years']
            
            # Remove original column since we have both score and standardized response
            df = df.drop('self_in_10_years', axis=1)
            
        return df

    def _handle_reform_choices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process reform choices into binary columns."""
        if 'preferred_reform_areas' not in df.columns:
            return df
            
        reform_choices = [
            'Quality Education',
            'Affordable Healthcare',
            'Better Infrastructure',
            'Smart Agriculture',
            'Tech Innovation',
            'Climate Action',
            'Job Creation'
        ]
        
        # Create binary columns for each reform choice
        for choice in reform_choices:
            col_name = f"reform_{choice.lower().replace(' ', '_')}"
            df[col_name] = df['preferred_reform_areas'].str.contains(
                choice, case=False
            ).astype(int)
        
        # Calculate total reforms selected
        reform_cols = [
            f"reform_{choice.lower().replace(' ', '_')}" 
            for choice in reform_choices
        ]
        df['num_reforms_selected'] = df[reform_cols].sum(axis=1)
        
        # Remove original column since we have individual binary columns
        df = df.drop('preferred_reform_areas', axis=1)
        
        return df

    def _format_currency(self, value: str) -> float:
        """Convert currency strings to float."""
        if pd.isna(value):
            return None
        try:
            # Remove any currency symbols and commas
            cleaned = re.sub(r'[^\d.]', '', value)
            return float(cleaned)
        except ValueError:
            print(f"Warning: Unable to parse income value '{value}'")
            return None

    def transform_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point for value transformation."""
        print("Starting value transformation...")
        
        df_transformed = df.copy()
        
        # Define all scale transformations with correct output column names
        scale_transformations = {
            'economic_freedom_view': 'economic_freedom_score',
            'cultural_festival_importance': 'cultural_importance_score',
            'justice_public_events_opinion': 'protest_support_score',
            'july_revolution_leadership_satisfaction': 'leadership_satisfaction_score',
            'tradition_based_decisions': 'tradition_reliance_score'
        }
        
        # Drop unwanted columns
        for col in self.columns_to_drop:
            if col in df_transformed.columns:
                df_transformed = df_transformed.drop(col, axis=1)
        
        # Transform phone numbers
        df_transformed = self._transform_phone_numbers(df_transformed)
        
        # Transform all scale columns
        for input_col, score_col in scale_transformations.items():
            df_transformed = self._transform_scale_column(
                df_transformed, input_col, score_col
            )
        
        # Handle other transformations
        df_transformed = self._transform_binary_choices(df_transformed)
        df_transformed = self._transform_categorical(df_transformed)
        df_transformed = self._handle_reform_choices(df_transformed)
        
        # Transform new binary question
        if 'political_change_needed' in df_transformed.columns:
            df_transformed['political_change_needed'] = (
                df_transformed['political_change_needed']
                .apply(self._clean_text_response)
                .map(self.binary_map)
            )
        
        # Transform family monthly income
        if 'family_monthly_income' in df_transformed.columns:
            df_transformed['family_monthly_income'] = df_transformed['family_monthly_income'].apply(self._format_currency)
        
        print("Value transformation completed!")
        return df_transformed 