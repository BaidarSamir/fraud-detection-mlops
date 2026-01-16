"""
Feature Engineering Module for Fraud Detection Pipeline

This module provides comprehensive feature engineering functions for the
IEEE-CIS Fraud Detection dataset. Features are designed to capture
fraud patterns while being robust for production deployment.

Author: Fraud Detection MLOps Team
Date: 2026-01-16
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    
    Implements a comprehensive feature engineering strategy including:
    - Missing value handling with intelligent imputation
    - Temporal feature extraction
    - Categorical encoding (label and frequency)
    - Interaction features
    - Aggregation features
    
    Attributes:
        random_seed: Random seed for reproducibility
        imputers: Dictionary of imputation values
        encoders: Dictionary of encoders
        aggregations: Dictionary of aggregation statistics
    """
    
    def __init__(self, random_seed: int = 42, drop_missing_threshold: float = 0.90):
        """
        Initialize FeatureEngineer.
        
        Args:
            random_seed: Random seed for reproducibility
            drop_missing_threshold: Threshold for dropping columns with high missing values
        """
        self.random_seed = random_seed
        self.drop_missing_threshold = drop_missing_threshold
        np.random.seed(random_seed)
        
        # Artifacts for inference
        self.imputers: Dict[str, Dict] = {}
        self.encoders: Dict[str, Any] = {}
        self.freq_maps: Dict[str, Dict] = {}
        self.aggregations: Dict[str, Dict] = {}
        self.feature_cols: List[str] = []
        
        # Define feature groups
        self.categorical_cols = [
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'DeviceType', 'DeviceInfo'
        ] + [f'id_{i:02d}' for i in range(12, 39)]
        
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Fit feature engineering on training data and transform both train and test.
        
        This method learns all necessary statistics from training data and
        applies transformations to both datasets. This ensures no data leakage.
        
        Args:
            train_df: Training DataFrame
            test_df: Optional test DataFrame
            
        Returns:
            Transformed DataFrame(s)
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Handle missing values
        logger.info(f"Handling missing values (threshold: {self.drop_missing_threshold})...")
        train_df, test_df = self._handle_missing_values(
            train_df, test_df, drop_threshold=self.drop_missing_threshold
        )
        
        # Create temporal features
        logger.info("Creating temporal features...")
        train_df = self._create_temporal_features(train_df)
        if test_df is not None:
            test_df = self._create_temporal_features(test_df)
        
        # Encode categorical features
        logger.info("Encoding categorical features...")
        train_df, test_df = self._encode_categorical_features(train_df, test_df)
        
        # Create frequency encoding
        logger.info("Creating frequency encodings...")
        train_df, test_df = self._create_frequency_encoding(train_df, test_df)
        
        # Create interaction features
        logger.info("Creating interaction features...")
        train_df = self._create_interaction_features(train_df)
        if test_df is not None:
            test_df = self._create_interaction_features(test_df)
        
        # Create aggregation features
        logger.info("Creating aggregation features...")
        train_df, test_df = self._create_aggregation_features(train_df, test_df)
        
        # Remove low variance features
        logger.info("Removing low variance features...")
        train_df, test_df = self._remove_low_variance_features(train_df, test_df)
        
        # Store feature columns
        exclude_cols = ['TransactionID', 'isFraud', 'TransactionDT']
        self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_cols)}")
        
        if test_df is not None:
            return train_df, test_df
        return train_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature engineering pipeline.
        
        Use this method for inference on new data after fitting on training data.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.imputers:
            raise ValueError("FeatureEngineer has not been fitted. Call fit_transform first.")
        
        logger.info("Transforming data with fitted pipeline...")
        
        # Apply imputation
        df = self._apply_imputation(df)
        
        # Create temporal features
        df = self._create_temporal_features(df)
        
        # Apply encoders
        df = self._apply_encoders(df)
        
        # Apply frequency encoding
        df = self._apply_frequency_encoding(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Apply aggregations
        df = self._apply_aggregations(df)
        
        return df
    
    def _handle_missing_values(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        drop_threshold: float = 0.90
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Handle missing values with intelligent imputation.
        
        Strategy:
        - Drop columns with >90% missing (little predictive value)
        - Create missing indicators for important columns
        - Impute numerical with median (robust to outliers)
        - Impute categorical with mode or 'missing' category
        """
        # Calculate missing percentages from training data
        missing_pct = train_df.isnull().sum() / len(train_df)
        
        # Identify columns to drop
        cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()
        cols_to_drop = [c for c in cols_to_drop if c not in ['TransactionID', 'isFraud']]
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{drop_threshold*100}% missing")
            train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
            if test_df is not None:
                test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
        
        # Create missing indicators for columns with significant missing (>10%)
        indicator_cols = missing_pct[(missing_pct > 0.10) & (missing_pct <= drop_threshold)].index.tolist()
        indicator_cols = [c for c in indicator_cols if c in train_df.columns][:20]
        
        for col in indicator_cols:
            train_df[f'{col}_missing'] = train_df[col].isnull().astype(np.int8)
            if test_df is not None:
                test_df[f'{col}_missing'] = test_df[col].isnull().astype(np.int8)
        
        # Get current categorical and numerical columns
        categorical_cols = [c for c in self.categorical_cols if c in train_df.columns]
        numerical_cols = [c for c in train_df.columns 
                         if c not in categorical_cols + ['TransactionID', 'isFraud']]
        
        # Impute numerical columns with median
        for col in numerical_cols:
            if train_df[col].isnull().sum() > 0:
                median_val = train_df[col].median()
                self.imputers[col] = {'strategy': 'median', 'value': float(median_val) if pd.notna(median_val) else 0}
                train_df[col] = train_df[col].fillna(self.imputers[col]['value'])
                if test_df is not None and col in test_df.columns:
                    test_df[col] = test_df[col].fillna(self.imputers[col]['value'])
        
        # Impute categorical columns
        for col in categorical_cols:
            if col in train_df.columns and train_df[col].isnull().sum() > 0:
                if train_df[col].dtype == 'object':
                    fill_val = 'missing'
                else:
                    fill_val = -999
                
                self.imputers[col] = {'strategy': 'constant', 'value': fill_val}
                train_df[col] = train_df[col].fillna(fill_val)
                if test_df is not None and col in test_df.columns:
                    test_df[col] = test_df[col].fillna(fill_val)
        
        return train_df, test_df
    
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned imputation to new data."""
        for col, imputer in self.imputers.items():
            if col in df.columns:
                df[col] = df[col].fillna(imputer['value'])
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from TransactionDT.
        
        TransactionDT is seconds from a reference datetime.
        We extract cyclical time features that capture fraud patterns.
        """
        if 'TransactionDT' not in df.columns:
            logger.warning("TransactionDT not found, skipping temporal features")
            return df
        
        df = df.copy()
        
        # Basic time extractions
        df['hour'] = (df['TransactionDT'] // 3600) % 24
        df['day'] = df['TransactionDT'] // (24 * 3600)
        df['day_of_week'] = df['day'] % 7
        df['day_of_month'] = df['day'] % 30
        
        # Cyclical encoding (captures that 23:00 is close to 00:00)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Time-based flags
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(np.int8)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                   (df['day_of_week'] < 5)).astype(np.int8)
        
        return df
    
    def _encode_categorical_features(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Label encode categorical features."""
        categorical_cols = [c for c in self.categorical_cols if c in train_df.columns]
        
        for col in categorical_cols:
            le = LabelEncoder()
            
            # Combine train and test for fitting (only if column exists in both)
            if test_df is not None and col in test_df.columns:
                combined = pd.concat([train_df[col].astype(str), 
                                     test_df[col].astype(str)], axis=0)
            else:
                combined = train_df[col].astype(str)
            
            le.fit(combined)
            self.encoders[col] = le
            
            train_df[col] = le.transform(train_df[col].astype(str))
            if test_df is not None and col in test_df.columns:
                test_df[col] = le.transform(test_df[col].astype(str))
        
        return train_df, test_df
    
    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned encoders to new data."""
        for col, encoder in self.encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        return df
    
    def _create_frequency_encoding(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Create frequency encoding for high-cardinality features."""
        # Select high-cardinality columns
        high_card_cols = [col for col in self.categorical_cols 
                         if col in train_df.columns and train_df[col].nunique() > 10][:10]
        
        for col in high_card_cols:
            freq_map = train_df[col].value_counts(normalize=True).to_dict()
            self.freq_maps[col] = freq_map
            
            train_df[f'{col}_freq'] = train_df[col].map(freq_map).fillna(0).astype(np.float32)
            if test_df is not None and col in test_df.columns:
                test_df[f'{col}_freq'] = test_df[col].map(freq_map).fillna(0).astype(np.float32)
        
        return train_df, test_df
    
    def _apply_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned frequency encoding to new data."""
        for col, freq_map in self.freq_maps.items():
            if col in df.columns:
                df[f'{col}_freq'] = df[col].map(freq_map).fillna(0).astype(np.float32)
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features capturing fraud patterns."""
        df = df.copy()
        
        # Transaction amount features
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
            df['TransactionAmt_decimal'] = (df['TransactionAmt'] - 
                                            np.floor(df['TransactionAmt'])).astype(np.float32)
            df['is_round_amount'] = (df['TransactionAmt_decimal'] < 0.01).astype(np.int8)
        
        # Email match (purchaser vs recipient)
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(np.int8)
        
        return df
    
    def _create_aggregation_features(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        group_cols: List[str] = ['card1', 'card2', 'addr1']
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Create aggregation features for transaction patterns."""
        for col in group_cols:
            if col not in train_df.columns:
                continue
            
            # Calculate aggregations on training data
            agg = train_df.groupby(col).agg({
                'TransactionAmt': ['count', 'mean', 'std']
            })
            agg.columns = [f'{col}_TransactionAmt_{stat}' for stat in ['count', 'mean', 'std']]
            agg = agg.reset_index()
            
            # Store for inference
            self.aggregations[col] = agg.set_index(col).to_dict('index')
            
            # Merge aggregations
            train_df = train_df.merge(agg, on=col, how='left')
            if test_df is not None:
                test_df = test_df.merge(agg, on=col, how='left')
            
            # Fill missing with global statistics
            for stat_col in agg.columns:
                if stat_col != col:
                    global_val = train_df[stat_col].median()
                    train_df[stat_col] = train_df[stat_col].fillna(global_val).astype(np.float32)
                    if test_df is not None:
                        test_df[stat_col] = test_df[stat_col].fillna(global_val).astype(np.float32)
        
        return train_df, test_df
    
    def _apply_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned aggregations to new data."""
        for col, agg_dict in self.aggregations.items():
            if col not in df.columns:
                continue
            
            for key, stats in list(agg_dict.items())[:1]:  # Get stat names from first entry
                for stat_name in stats.keys():
                    col_name = f'{col}_TransactionAmt_{stat_name.split("_")[-1]}'
                    df[col_name] = df[col].map(lambda x: agg_dict.get(x, {}).get(stat_name, 0))
        
        return df
    
    def _remove_low_variance_features(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Remove features with variance below threshold."""
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [c for c in numerical_cols if c not in ['TransactionID', 'isFraud']]
        
        variances = train_df[numerical_cols].var()
        low_var_cols = variances[variances < threshold].index.tolist()
        
        if low_var_cols:
            logger.info(f"Removing {len(low_var_cols)} low variance features")
            train_df = train_df.drop(columns=low_var_cols)
            if test_df is not None:
                test_df = test_df.drop(columns=low_var_cols, errors='ignore')
        
        return train_df, test_df
    
    def save_artifacts(self, path: Union[str, Path]) -> None:
        """
        Save feature engineering artifacts for inference.
        
        Args:
            path: Path to save artifacts
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            'imputers': self.imputers,
            'encoders': self.encoders,
            'freq_maps': self.freq_maps,
            'aggregations': self.aggregations,
            'feature_cols': self.feature_cols,
            'categorical_cols': self.categorical_cols,
            'random_seed': self.random_seed,
            'drop_missing_threshold': self.drop_missing_threshold
        }
        
        with open(path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Feature engineering artifacts saved to {path}")
    
    def load_artifacts(self, path: Union[str, Path]) -> None:
        """
        Load feature engineering artifacts for inference.
        
        Args:
            path: Path to load artifacts from
        """
        with open(path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.imputers = artifacts['imputers']
        self.encoders = artifacts['encoders']
        self.freq_maps = artifacts['freq_maps']
        self.aggregations = artifacts['aggregations']
        self.feature_cols = artifacts['feature_cols']
        self.categorical_cols = artifacts['categorical_cols']
        self.random_seed = artifacts['random_seed']
        self.drop_missing_threshold = artifacts.get('drop_missing_threshold', 0.90)
        
        logger.info(f"Feature engineering artifacts loaded from {path}")


def get_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features into groups for analysis.
    
    Args:
        feature_cols: List of feature column names
        
    Returns:
        Dictionary mapping group names to feature lists
    """
    groups = {
        'Vesta Features': [],
        'Counting Features': [],
        'Timedelta Features': [],
        'Match Features': [],
        'Identity Features': [],
        'Card Features': [],
        'Address Features': [],
        'Email Features': [],
        'Temporal Features': [],
        'Amount Features': [],
        'Other Features': []
    }
    
    temporal_features = ['hour', 'day', 'day_of_week', 'hour_sin', 'hour_cos',
                        'dow_sin', 'dow_cos', 'is_night', 'is_weekend', 'is_business_hours']
    
    for col in feature_cols:
        if col.startswith('V'):
            groups['Vesta Features'].append(col)
        elif col.startswith('C') and col[1:].replace('_', '').isdigit():
            groups['Counting Features'].append(col)
        elif col.startswith('D') and col[1:].replace('_', '').isdigit():
            groups['Timedelta Features'].append(col)
        elif col.startswith('M') and col[1:].replace('_', '').isdigit():
            groups['Match Features'].append(col)
        elif col.startswith('id_'):
            groups['Identity Features'].append(col)
        elif col.startswith('card'):
            groups['Card Features'].append(col)
        elif col.startswith('addr'):
            groups['Address Features'].append(col)
        elif 'email' in col.lower():
            groups['Email Features'].append(col)
        elif col in temporal_features or col.startswith('day'):
            groups['Temporal Features'].append(col)
        elif 'TransactionAmt' in col:
            groups['Amount Features'].append(col)
        else:
            groups['Other Features'].append(col)
    
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}
    
    return groups


if __name__ == '__main__':
    # Example usage
    print("Feature Engineering Module")
    print("Usage: from feature_engineering import FeatureEngineer")
