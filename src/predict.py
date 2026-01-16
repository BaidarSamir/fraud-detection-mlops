"""
Inference Pipeline for Fraud Detection Model

This module provides production-ready inference capabilities including:
- Real-time single transaction scoring
- Batch prediction
- Model monitoring and drift detection
- Databricks-compatible deployment patterns

Author: Fraud Detection MLOps Team
Date: 2026-01-16
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPredictor:
    """
    Production inference pipeline for fraud detection.
    
    Features:
    - Single transaction and batch prediction
    - Configurable threshold for precision-recall trade-off
    - Prediction monitoring and logging
    - Databricks/Spark compatible design
    
    Attributes:
        model: Trained model
        threshold: Classification threshold
        feature_cols: List of feature columns
        feature_engineer: Feature engineering pipeline
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_artifacts_path: Optional[str] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize FraudDetectionPredictor.
        
        Args:
            model_path: Path to saved model artifacts
            feature_artifacts_path: Path to feature engineering artifacts
            threshold: Override threshold (uses optimal if None)
        """
        self.model = None
        self.model_name = None
        self.threshold = threshold
        self.feature_cols: List[str] = []
        self.feature_engineer = None
        
        # Monitoring state
        self.prediction_count = 0
        self.fraud_predictions = 0
        self.prediction_times: List[float] = []
        self.prediction_distribution: List[float] = []
        
        if model_path:
            self.load_model(model_path)
        
        if feature_artifacts_path:
            self.load_feature_artifacts(feature_artifacts_path)
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Path to model pickle file
        """
        logger.info(f"Loading model from {path}")
        
        with open(path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.model = artifacts['best_model']
        self.model_name = artifacts['best_model_name']
        self.feature_cols = artifacts['feature_cols']
        
        # Use saved threshold if not overridden
        if self.threshold is None:
            self.threshold = artifacts.get('optimal_threshold', 0.5)
        
        logger.info(f"Model loaded: {self.model_name}, threshold: {self.threshold:.4f}")
    
    def load_feature_artifacts(self, path: str) -> None:
        """
        Load feature engineering artifacts for preprocessing.
        
        Args:
            path: Path to feature artifacts pickle file
        """
        from feature_engineering import FeatureEngineer
        
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.load_artifacts(path)
        
        logger.info("Feature engineering artifacts loaded")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get fraud probability predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure correct feature columns
        available_features = [c for c in self.feature_cols if c in X.columns]
        if len(available_features) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_features)
            logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
        
        X_subset = X[available_features]
        
        return self.model.predict_proba(X_subset)[:, 1]
    
    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Get binary fraud predictions.
        
        Args:
            X: Feature DataFrame
            threshold: Optional threshold override
            
        Returns:
            Array of binary predictions (0=legitimate, 1=fraud)
        """
        threshold = threshold or self.threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def predict_with_details(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get predictions with probability scores and confidence.
        
        Args:
            X: Feature DataFrame
            threshold: Optional threshold override
            
        Returns:
            DataFrame with predictions, probabilities, and confidence
        """
        threshold = threshold or self.threshold
        
        start_time = time.time()
        proba = self.predict_proba(X)
        prediction_time = time.time() - start_time
        
        predictions = (proba >= threshold).astype(int)
        
        # Calculate confidence (distance from threshold)
        confidence = np.abs(proba - threshold) / max(threshold, 1 - threshold)
        
        result = pd.DataFrame({
            'fraud_probability': proba,
            'is_fraud': predictions,
            'confidence': confidence,
            'threshold_used': threshold
        })
        
        # Update monitoring stats
        self._update_monitoring(proba, predictions, prediction_time)
        
        return result
    
    def predict_single(
        self,
        transaction: Dict[str, Any],
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.
        
        This method is designed for real-time inference scenarios.
        
        Args:
            transaction: Dictionary with transaction features
            preprocess: Whether to apply feature engineering
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Apply feature engineering if available and requested
        if preprocess and self.feature_engineer is not None:
            df = self.feature_engineer.transform(df)
        
        # Get prediction
        proba = self.predict_proba(df)[0]
        is_fraud = int(proba >= self.threshold)
        
        inference_time = time.time() - start_time
        
        result = {
            'transaction_id': transaction.get('TransactionID', 'unknown'),
            'fraud_probability': float(proba),
            'is_fraud': is_fraud,
            'threshold': self.threshold,
            'confidence': float(abs(proba - self.threshold) / max(self.threshold, 1 - self.threshold)),
            'model_name': self.model_name,
            'inference_time_ms': inference_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update monitoring
        self._update_monitoring(np.array([proba]), np.array([is_fraud]), inference_time)
        
        return result
    
    def predict_batch(
        self,
        transactions: pd.DataFrame,
        preprocess: bool = True,
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """
        Batch prediction for multiple transactions.
        
        Processes data in chunks for memory efficiency.
        
        Args:
            transactions: DataFrame with transactions
            preprocess: Whether to apply feature engineering
            batch_size: Number of transactions per batch
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Processing {len(transactions)} transactions in batches of {batch_size}")
        
        all_results = []
        
        for i in range(0, len(transactions), batch_size):
            batch = transactions.iloc[i:i + batch_size]
            
            if preprocess and self.feature_engineer is not None:
                batch = self.feature_engineer.transform(batch.copy())
            
            results = self.predict_with_details(batch)
            
            # Add transaction IDs if available
            if 'TransactionID' in transactions.columns:
                results['TransactionID'] = batch['TransactionID'].values
            
            all_results.append(results)
            
            logger.info(f"Processed batch {i//batch_size + 1}: "
                       f"{len(results)} transactions, "
                       f"{results['is_fraud'].sum()} flagged as fraud")
        
        return pd.concat(all_results, ignore_index=True)
    
    def _update_monitoring(
        self,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        inference_time: float
    ) -> None:
        """Update monitoring statistics."""
        self.prediction_count += len(predictions)
        self.fraud_predictions += predictions.sum()
        self.prediction_times.append(inference_time)
        self.prediction_distribution.extend(probabilities.tolist())
        
        # Keep only recent predictions for distribution
        if len(self.prediction_distribution) > 10000:
            self.prediction_distribution = self.prediction_distribution[-10000:]
    
    def get_monitoring_stats(self) -> Dict:
        """
        Get monitoring statistics for the predictor.
        
        Returns:
            Dictionary with monitoring metrics
        """
        if not self.prediction_distribution:
            return {'status': 'no predictions yet'}
        
        proba_array = np.array(self.prediction_distribution)
        
        return {
            'total_predictions': self.prediction_count,
            'fraud_predictions': self.fraud_predictions,
            'fraud_rate': self.fraud_predictions / max(self.prediction_count, 1),
            'avg_inference_time_ms': np.mean(self.prediction_times) * 1000 if self.prediction_times else 0,
            'probability_distribution': {
                'mean': float(np.mean(proba_array)),
                'std': float(np.std(proba_array)),
                'median': float(np.median(proba_array)),
                'p95': float(np.percentile(proba_array, 95)),
                'p99': float(np.percentile(proba_array, 99))
            },
            'threshold': self.threshold,
            'model_name': self.model_name
        }
    
    def check_feature_drift(
        self,
        reference_stats: Dict[str, Dict],
        current_data: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict:
        """
        Check for feature drift between reference and current data.
        
        Feature drift indicates that the model may need retraining.
        
        Args:
            reference_stats: Dictionary of reference feature statistics
            current_data: Current data to check
            threshold: Threshold for drift detection
            
        Returns:
            Dictionary with drift analysis
        """
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features_checked': 0,
            'features_drifted': 0,
            'drift_details': {}
        }
        
        for feature, ref_stats in reference_stats.items():
            if feature not in current_data.columns:
                continue
            
            drift_report['features_checked'] += 1
            
            current_mean = current_data[feature].mean()
            current_std = current_data[feature].std()
            
            ref_mean = ref_stats.get('mean', 0)
            ref_std = ref_stats.get('std', 1)
            
            # Calculate normalized difference
            mean_diff = abs(current_mean - ref_mean) / (ref_std + 1e-8)
            
            if mean_diff > threshold:
                drift_report['features_drifted'] += 1
                drift_report['drift_details'][feature] = {
                    'reference_mean': ref_mean,
                    'current_mean': float(current_mean),
                    'normalized_difference': float(mean_diff)
                }
        
        drift_report['drift_detected'] = drift_report['features_drifted'] > 0
        
        return drift_report


class DatabricksModelWrapper:
    """
    Wrapper for deploying fraud detection model on Databricks.
    
    This class provides an interface compatible with Databricks Model Serving
    and MLflow model registry.
    """
    
    def __init__(self, predictor: FraudDetectionPredictor):
        """
        Initialize Databricks wrapper.
        
        Args:
            predictor: FraudDetectionPredictor instance
        """
        self.predictor = predictor
    
    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict method compatible with MLflow and Databricks.
        
        Args:
            model_input: Input DataFrame
            
        Returns:
            DataFrame with predictions
        """
        return self.predictor.predict_with_details(model_input)
    
    def predict_udf(self):
        """
        Create a Pandas UDF for Spark DataFrame scoring.
        
        Usage in Databricks:
            predict_fn = model.predict_udf()
            predictions = df.withColumn('fraud_score', predict_fn(struct(*feature_cols)))
        """
        try:
            from pyspark.sql.functions import pandas_udf
            from pyspark.sql.types import DoubleType
            
            model = self.predictor.model
            feature_cols = self.predictor.feature_cols
            
            @pandas_udf(DoubleType())
            def score_transaction(features: pd.DataFrame) -> pd.Series:
                available = [c for c in feature_cols if c in features.columns]
                proba = model.predict_proba(features[available])[:, 1]
                return pd.Series(proba)
            
            return score_transaction
        except ImportError:
            logger.warning("PySpark not available. UDF creation skipped.")
            return None


def create_mock_scoring_function(model_path: str, feature_artifacts_path: str):
    """
    Create a mock real-time scoring function for testing.
    
    This simulates how the model would be called in a production
    REST API or streaming context.
    
    Args:
        model_path: Path to model artifacts
        feature_artifacts_path: Path to feature artifacts
        
    Returns:
        Scoring function
    """
    predictor = FraudDetectionPredictor(
        model_path=model_path,
        feature_artifacts_path=feature_artifacts_path
    )
    
    def score_transaction(transaction_json: str) -> str:
        """
        Score a single transaction from JSON input.
        
        Args:
            transaction_json: JSON string with transaction data
            
        Returns:
            JSON string with prediction result
        """
        transaction = json.loads(transaction_json)
        result = predictor.predict_single(transaction, preprocess=True)
        return json.dumps(result)
    
    return score_transaction


def main():
    """Example inference script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run fraud detection inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model artifacts')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data for prediction')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save predictions')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Classification threshold')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FraudDetectionPredictor(
        model_path=args.model_path,
        threshold=args.threshold
    )
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    if args.data_path.endswith('.parquet'):
        data = pd.read_parquet(args.data_path)
    else:
        data = pd.read_csv(args.data_path)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = predictor.predict_batch(data, preprocess=False)
    
    # Save predictions
    predictions.to_csv(args.output_path, index=False)
    logger.info(f"Predictions saved to {args.output_path}")
    
    # Print monitoring stats
    stats = predictor.get_monitoring_stats()
    print("\nMonitoring Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
