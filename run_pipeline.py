"""
Complete MLOps Pipeline for Fraud Detection

This script orchestrates the entire lifecycle:
1. Data Loading & Validation (using DataProcessor)
2. Feature Engineering (using FeatureEngineer)
3. Model Training & Tracking (using FraudDetectionTrainer)
4. Inference (using FraudDetectionPredictor)

Usage:
    python run_pipeline.py --stage all
    python run_pipeline.py --stage data --check-quality
    python run_pipeline.py --stage train --experiment-name "v1_baseline"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
BASE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_PATH / 'src'))

from data_processing import DataProcessor, run_data_quality_checks
from feature_engineering import FeatureEngineer
from train import FraudDetectionTrainer
from predict import FraudDetectionPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_execution.log')
    ]
)
logger = logging.getLogger(__name__)

def run_data_stage(args):
    """Execute data processing stage."""
    logger.info("="*80)
    logger.info("STAGE: DATA PROCESSING")
    logger.info("="*80)
    
    data_path = BASE_PATH / 'Data' / 'raw'
    processed_path = BASE_PATH / 'Data' / 'processed'
    processed_path.mkdir(parents=True, exist_ok=True)
    
    processor = DataProcessor(data_path)
    
    # Load and merge
    df = processor.load_and_merge_data()
    
    # Validate
    validation = processor.validate_data(df)
    logger.info(f"Data validation: {validation}")
    
    if args.check_quality:
        quality = run_data_quality_checks(df)
        logger.info(f"Data quality checks: {quality}")
    
    return df

def run_features_stage(df, args):
    """Execute feature engineering stage."""
    logger.info("="*80)
    logger.info("STAGE: FEATURE ENGINEERING")
    logger.info("="*80)
    
    features_path = BASE_PATH / 'Data' / 'features'
    processed_path = BASE_PATH / 'Data' / 'processed'
    features_path.mkdir(parents=True, exist_ok=True)
    
    fe = FeatureEngineer(drop_missing_threshold=args.drop_missing_threshold)
    
    # Split train/test for processing if "isFraud" is present (it is in train)
    # For this pipeline, we assume we are running on the full training set
    # In a real scenario, we might handle train/test files separately
    
    # We will treat the loaded df as training data
    # If we had a separate test file, we would load it there.
    # For simplicity in this script, let's look for test file too.
    
    data_path = BASE_PATH / 'Data' / 'raw'
    test_transaction = data_path / 'test_transaction.csv'
    test_df = None
    
    if test_transaction.exists():
        logger.info("Loading test data for feature engineering...")
        processor = DataProcessor(data_path)
        # We need a slight modification to load test data properly 
        # using the processor implies we might need a separate method 
        # or just use load_and_merge with different filenames
        test_df = processor.load_and_merge_data(
            transaction_file='test_transaction.csv',
            identity_file='test_identity.csv'
        )
    
    train_df, test_df = fe.fit_transform(df, test_df)
    
    # Save processed data
    logger.info("Saving processed datasets...")
    train_df.to_parquet(processed_path / 'train_processed.parquet', index=False)
    if test_df is not None:
        test_df.to_parquet(processed_path / 'test_processed.parquet', index=False)
        
    # Save artifacts
    fe.save_artifacts(features_path / 'feature_artifacts.pkl')
    
    return train_df, test_df

def run_train_stage(train_df, args):
    """Execute training stage."""
    logger.info("="*80)
    logger.info("STAGE: MODEL TRAINING")
    logger.info("="*80)
    
    output_path = BASE_PATH / 'outputs' / 'models' / 'model.pkl'
    features_path = BASE_PATH / 'Data' / 'features'
    
    # Feature Engineering Artifacts needed for column names
    import pickle
    with open(features_path / 'feature_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    feature_cols = artifacts['feature_cols']
    
    # Prepare data
    logger.info("Preparing training data...")
    # Time-based split
    train_df_sorted = train_df.sort_values('TransactionDT').reset_index(drop=True)
    split_idx = int(len(train_df_sorted) * 0.8)
    
    train_data = train_df_sorted.iloc[:split_idx]
    val_data = train_df_sorted.iloc[split_idx:]
    
    available_features = [c for c in feature_cols if c in train_data.columns]
    
    X_train = train_data[available_features]
    y_train = train_data['isFraud'].values
    X_val = val_data[available_features]
    y_val = val_data['isFraud'].values
    
    trainer = FraudDetectionTrainer(experiment_name=args.experiment_name)
    
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    trainer.save_model(output_path, available_features)
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description='Fraud Detection MLOps Pipeline')
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['data', 'features', 'train', 'all'],
                       help='Pipeline stage to run')
    parser.add_argument('--check-quality', action='store_true',
                       help='Run data quality checks')
    parser.add_argument('--experiment-name', type=str, default='fraud_detection_v1',
                       help='MLflow experiment name')
    parser.add_argument('--drop-missing-threshold', type=float, default=0.90,
                       help='Threshold for dropping columns with high missing values (default: 0.90)')
    
    args = parser.parse_args()
    
    try:
        if args.stage in ['data', 'all']:
            # Load Raw Data
            raw_df = run_data_stage(args)
            
        if args.stage in ['features', 'all']:
            # If skipping data stage, load raw data manually (or implemented loading from saved raw step if needed)
            # For 'features' only, we assume raw_df is passed or we load it. 
            # Ideally, 'features' only should load from raw files again.
            if args.stage == 'features':
                 raw_df = run_data_stage(args) # Re-run data load for safety
            
            train_processed, test_processed = run_features_stage(raw_df, args)
            
        if args.stage in ['train', 'all']:
            # If just training, load processed data
            if args.stage == 'train':
                import pandas as pd
                processed_path = BASE_PATH / 'Data' / 'processed' / 'train_processed.parquet'
                if not processed_path.exists():
                    raise FileNotFoundError("Processed data not found. Run 'features' stage first.")
                train_processed = pd.read_parquet(processed_path)
            
            run_train_stage(train_processed, args)
            
        logger.info("\nPipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
