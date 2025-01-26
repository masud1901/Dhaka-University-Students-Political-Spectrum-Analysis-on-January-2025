"""Module for normalizing column names in the DU student survey data."""
import pandas as pd


class ColumnNormalizer:
    """Handles column name standardization for survey data."""
    
    def __init__(self):
        """Initialize column mapping dictionaries."""
        # Basic columns
        self.basic_columns = {
            'Timestamp': 'timestamp',
            'Name': 'name',
            'Phone number': 'phone',
            'Your program': 'program',
            'Session': 'session',
            "My Family's monthly income": 'family_monthly_income',
            "Do you think Bangladesh badly needs political change?": 'political_change_needed'
        }
        
        # Special case for economic freedom question
        self.economic_freedom_question = (
            '"Economic growth is best achieved when the government minimizes '
            'regulations and allows businesses to operate freely."  '
            '[ ব্যবসা প্রতিষ্ঠানকে স্বাধীনভাবে চলতে দিলে/সরকারি নিয়ন্ত্রণ '
            'কমালে অর্থনৈতিক সমৃদ্ধি সবচেয়ে ভালো হয়]'
        )
        
        # Long form questions with Bengali translations
        self.bilingual_questions = [
            (
                'Which reform initiatives are you most interested in? '
                '(choose 3) [ উন্নয়ন উদ্যোগে কোন খাতে আপনি বেশি আগ্রহী?]',
                'preferred_reform_areas'
            ),
            (
                'How satisfied are you with the post July revolution '
                'leadership? [আপনি জুলাই বিপ্লব পরবর্তী নেতৃত্বে কতটুকু সন্তুষ্ট?]',
                'july_revolution_leadership_satisfaction'
            ),
            (
                'Cultural festivals and rituals (nababarsha, puja) are '
                'essential for maintaining social cohesion. [ সাংস্কৃতিক '
                'উৎসব ও রীতি (নববর্ষ, পূজা) সামাজিক সংহতি বজায় রাখতে '
                'অত্যন্ত গুরুত্বপূর্ণ]',
                'cultural_festival_importance'
            )
        ]
        
        # English-only questions
        self.english_questions = {
            'How would you like to see yourself in 10 years?': 'self_in_10_years',
            ('How often do you rely on the wisdom of elders or long-standing '
             'traditions when making decisions?'): 'tradition_based_decisions',
            'Which traditional Bangladeshi food best represents the nation?': 
                'traditional_food_representation',
            ('How do you feel about public celebrations or protests that '
             'emphasize equity and justice?'): 'justice_public_events_opinion'
        }
        
        # Binary choice questions
        self.binary_questions = [
            (
                'Choose 1 from each pair  [Economic growth over Protecting '
                'the environment?]',
                'economic_growth_over_environment'
            ),
            (
                'Choose 1 from each pair  [Individual freedom over Social '
                'harmony?]',
                'individual_freedom_over_harmony'
            ),
            (
                'Choose 1 from each pair  [Traditional values over '
                'Progressive reforms?]',
                'traditional_over_progressive'
            ),
            (
                'Choose 1 from each pair  [Central authority over Local '
                'autonomy?]',
                'central_over_local'
            ),
            (
                'Choose 1 from each pair  [Market freedom over Government '
                'regulation?]',
                'market_over_regulation'
            ),
            (
                'Choose 1 from each pair  [Social stability over Justice?]',
                'stability_over_justice'
            )
        ]
        
        # Create combined mapping dictionaries
        self.column_map = {**self.basic_columns, **self.english_questions}
        self.column_map.update(dict(self.bilingual_questions))
        # Add special case for economic freedom question
        self.column_map[self.economic_freedom_question] = 'economic_freedom_view'
        self.binary_pairs = dict(self.binary_questions)

    def _clean_basic_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize basic column names."""
        cleaned_cols = []
        for col in df.columns:
            # Special handling for economic freedom question
            if 'Economic growth is best achieved' in col:
                cleaned_cols.append(self.economic_freedom_question)
                continue
                
            # Regular cleaning for other columns
            clean_col = (
                str(col)
                .strip()
                .replace('""', '"')  # Fix double quotes
                .replace('\n', ' ')  # Replace newlines with space
            )
            cleaned_cols.append(clean_col)
        
        df.columns = cleaned_cols
        return df

    def _apply_column_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standardized column mappings."""
        return df.rename(columns=self.column_map)

    def _handle_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle binary choice column names."""
        return df.rename(columns=self.binary_pairs)

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point for column normalization."""
        print("Starting column name normalization...")
        
        # Step 1: Basic cleaning (handle whitespace and quotes)
        df = self._clean_basic_column_names(df)
        
        # Step 2: Handle binary choice columns first
        df = self._handle_binary_columns(df)
        
        # Step 3: Apply other column mappings
        df = self._apply_column_mappings(df)
        
        print("Column normalization completed!")
        return df 