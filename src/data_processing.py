"""
Data Processing Module for Fraud Detection Pipeline

This module provides reusable functions for loading, merging, and validating
the IEEE-CIS Fraud Detection dataset. Designed for production deployment
on Databricks with Delta Lake compatibility.

Author: Fraud Detection MLOps Team
Date: 2026-01-16
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processor for fraud detection pipeline.
    
    Handles data loading, merging, validation, and memory optimization.
    Designed to work with both local files and Databricks Delta tables.
    
    Attributes:
        data_path: Path to data directory
        random_seed: Random seed for reproducibility
    """
    
    def __init__(self, data_path: Union[str, Path], random_seed: int = 42):
        """
        Initialize DataProcessor.
        
        Args:
            data_path: Path to directory containing data files
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def load_and_merge_data(
        self,
        transaction_file: str = 'train_transaction.csv',
        identity_file: str = 'train_identity.csv',
        optimize_memory: bool = True
    ) -> pd.DataFrame:
        """
        Load and merge transaction and identity data.
        
        The merge is performed as a left join on TransactionID because
        not all transactions have associated identity information.
        Missing identity data itself can be a signal for fraud.
        
        Args:
            transaction_file: Name of transaction CSV file
            identity_file: Name of identity CSV file
            optimize_memory: Whether to optimize memory usage
            
        Returns:
            Merged DataFrame
            
        Raises:
            FileNotFoundError: If data files don't exist
        """
        logger.info("Loading transaction data...")
        transaction_path = self.data_path / transaction_file
        identity_path = self.data_path / identity_file
        
        if not transaction_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {transaction_path}")
        
        # Load transaction data
        transaction_df = pd.read_csv(transaction_path)
        logger.info(f"Transaction data loaded: {transaction_df.shape}")
        
        # Load identity data if exists
        if identity_path.exists():
            logger.info("Loading identity data...")
            identity_df = pd.read_csv(identity_path)
            logger.info(f"Identity data loaded: {identity_df.shape}")
            
            # Merge datasets
            logger.info("Merging datasets...")
            merged_df = transaction_df.merge(identity_df, on='TransactionID', how='left')
            
            # Log merge statistics
            identity_coverage = identity_df.shape[0] / transaction_df.shape[0] * 100
            logger.info(f"Identity coverage: {identity_coverage:.1f}%")
            
            del transaction_df, identity_df
        else:
            logger.warning(f"Identity file not found: {identity_path}")
            merged_df = transaction_df
            del transaction_df
        
        # Optimize memory if requested
        if optimize_memory:
            merged_df = self.reduce_memory_usage(merged_df)
        
        logger.info(f"Final dataset shape: {merged_df.shape}")
        return merged_df
    
    def reduce_memory_usage(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Reduce memory usage by downcasting numeric types.
        
        This is essential for production systems handling large datasets,
        particularly when deploying to Databricks or other cloud platforms.
        
        Args:
            df: Input DataFrame
            verbose: Whether to print memory reduction stats
            
        Returns:
            DataFrame with optimized dtypes
        """
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Downcast integers
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        
                # Downcast floats
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        
        if verbose:
            logger.info(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
                       f"({reduction:.1f}% reduction)")
        
        return df
    
    def validate_data(self, df: pd.DataFrame, is_training: bool = True) -> Dict:
        """
        Validate data quality and return summary statistics.
        
        Checks performed:
        - Missing value rates
        - Target variable distribution (training only)
        - Data type consistency
        - Outlier detection for key columns
        
        Args:
            df: DataFrame to validate
            is_training: Whether this is training data (has target column)
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data quality...")
        
        validation_results = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_rates': {},
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        # Calculate missing rates
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        validation_results['missing_rates'] = {
            'columns_with_missing': (missing_pct > 0).sum(),
            'columns_over_50_pct_missing': (missing_pct > 50).sum(),
            'columns_over_90_pct_missing': (missing_pct > 90).sum(),
            'top_missing': missing_pct.nlargest(10).to_dict()
        }
        
        # Validate target column for training data
        if is_training:
            if 'isFraud' not in df.columns:
                logger.error("Target column 'isFraud' not found in training data")
                validation_results['target_valid'] = False
            else:
                fraud_rate = df['isFraud'].mean()
                validation_results['target_valid'] = True
                validation_results['fraud_rate'] = fraud_rate
                validation_results['fraud_count'] = df['isFraud'].sum()
                validation_results['legitimate_count'] = (df['isFraud'] == 0).sum()
                
                # Check for reasonable fraud rate
                if fraud_rate < 0.001 or fraud_rate > 0.5:
                    logger.warning(f"Unusual fraud rate detected: {fraud_rate:.4f}")
        
        # Validate transaction amount
        if 'TransactionAmt' in df.columns:
            amt_stats = df['TransactionAmt'].describe()
            validation_results['transaction_amount'] = {
                'min': amt_stats['min'],
                'max': amt_stats['max'],
                'mean': amt_stats['mean'],
                'median': df['TransactionAmt'].median()
            }
            
            # Flag potential issues
            if amt_stats['min'] < 0:
                logger.warning("Negative transaction amounts detected")
        
        logger.info(f"Validation complete. Rows: {validation_results['n_rows']}, "
                   f"Columns: {validation_results['n_columns']}")
        
        return validation_results
    
    def split_data_by_time(
        self,
        df: pd.DataFrame,
        time_col: str = 'TransactionDT',
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by time for proper train/validation setup.
        
        Time-based splitting is critical for fraud detection because:
        1. Production models always predict on future transactions
        2. Random splits would leak future information
        3. Reveals concept drift in fraud patterns
        
        Args:
            df: DataFrame with time column
            time_col: Name of time column
            train_ratio: Proportion of data for training
            
        Returns:
            Tuple of (train_df, val_df)
        """
        if time_col not in df.columns:
            logger.warning(f"Time column '{time_col}' not found. Using random split.")
            from sklearn.model_selection import train_test_split
            return train_test_split(df, train_size=train_ratio, random_state=self.random_seed)
        
        # Sort by time
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(df_sorted) * train_ratio)
        
        train_df = df_sorted.iloc[:split_idx]
        val_df = df_sorted.iloc[split_idx:]
        
        logger.info(f"Time-based split: Train={len(train_df)}, Val={len(val_df)}")
        logger.info(f"Train fraud rate: {train_df['isFraud'].mean()*100:.2f}%")
        logger.info(f"Val fraud rate: {val_df['isFraud'].mean()*100:.2f}%")
        
        return train_df, val_df


def run_data_quality_checks(df: pd.DataFrame) -> Dict:
    """
    Run comprehensive data quality checks for monitoring.
    
    This function is designed to be called periodically in production
    to detect data drift and quality issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    checks = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'row_count': len(df),
        'null_percentage': df.isnull().sum().sum() / df.size * 100,
        'duplicate_transactions': df.duplicated(subset=['TransactionID']).sum() if 'TransactionID' in df.columns else 0
    }
    
    # Check for expected columns
    expected_cols = ['TransactionID', 'TransactionAmt', 'ProductCD', 'card1']
    checks['missing_columns'] = [c for c in expected_cols if c not in df.columns]
    
    # Statistical checks for key columns
    if 'TransactionAmt' in df.columns:
        amt = df['TransactionAmt']
        checks['transaction_amt_stats'] = {
            'mean': amt.mean(),
            'std': amt.std(),
            'min': amt.min(),
            'max': amt.max(),
            'null_count': amt.isnull().sum()
        }
    
    return checks


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '../Data/raw'
    
    processor = DataProcessor(data_path)
    
    # Load and validate data
    df = processor.load_and_merge_data()
    validation = processor.validate_data(df)
    
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
