"""
Training Pipeline for Fraud Detection Model

This module provides a production-ready training pipeline with MLflow
experiment tracking, cross-validation, and model selection.

Author: Fraud Detection MLOps Team
Date: 2026-01-16
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Import MLflow
try:
    import mlflow
    import mlflow.lightgbm
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Experiment tracking disabled.")

# Import gradient boosting libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """
    Training pipeline for fraud detection models.
    
    Features:
    - Multiple model comparison (LightGBM, XGBoost, Random Forest)
    - MLflow experiment tracking
    - Class imbalance handling
    - Time-based train/validation split
    - Automatic threshold optimization
    
    Attributes:
        random_seed: Random seed for reproducibility
        experiment_name: MLflow experiment name
        models: Dictionary of trained models
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        experiment_name: str = "fraud_detection",
        mlflow_tracking_uri: Optional[str] = None
    ):
        """
        Initialize FraudDetectionTrainer.
        
        Args:
            random_seed: Random seed for reproducibility
            experiment_name: Name for MLflow experiment
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.random_seed = random_seed
        self.experiment_name = experiment_name
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.best_threshold: float = 0.5
        
        np.random.seed(random_seed)
        
        # Configure MLflow
        if MLFLOW_AVAILABLE and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
    
    def calculate_class_weights(self, y: np.ndarray) -> Tuple[Dict, float]:
        """
        Calculate class weights for imbalanced classification.
        
        For fraud detection, the minority class (fraud) needs higher weight
        to ensure the model learns to detect it.
        
        Args:
            y: Target array
            
        Returns:
            Tuple of (class_weights dict, scale_pos_weight for boosting)
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        # scale_pos_weight for LightGBM/XGBoost
        scale_pos_weight = class_weights[1] / class_weights[0]
        
        logger.info(f"Class weights - 0: {class_weights[0]:.4f}, 1: {class_weights[1]:.4f}")
        logger.info(f"Scale pos weight: {scale_pos_weight:.4f}")
        
        return class_weights, scale_pos_weight
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive model evaluation for fraud detection.
        
        Args:
            y_true: Actual labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
            'f1': float(f1_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'threshold': threshold
        }
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold based on F1 score.
        
        Args:
            y_true: Actual labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (optimal_threshold, best_f1_score)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        best_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
        best_f1 = float(f1_scores[best_idx])
        
        return optimal_threshold, best_f1
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        scale_pos_weight: float,
        params: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Train LightGBM model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            scale_pos_weight: Weight for positive class
            params: Optional custom parameters
            
        Returns:
            Tuple of (model, metrics)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")
        
        default_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 64,
            'min_child_samples': 100,
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_seed,
            'n_jobs': -1,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        model = lgb.LGBMClassifier(**default_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        optimal_threshold, _ = self.find_optimal_threshold(y_val, y_pred_proba)
        metrics = self.evaluate_model(y_val, y_pred_proba, optimal_threshold)
        metrics['optimal_threshold'] = optimal_threshold
        
        return model, metrics
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        scale_pos_weight: float,
        params: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Train XGBoost model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            scale_pos_weight: Weight for positive class
            params: Optional custom parameters
            
        Returns:
            Tuple of (model, metrics)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        
        default_params = {
            'objective': 'binary:logistic',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_child_weight': 100,
            'scale_pos_weight': scale_pos_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_seed,
            'n_jobs': -1,
            'early_stopping_rounds': 100,
            'eval_metric': 'auc'
        }
        
        if params:
            default_params.update(params)
        
        model = xgb.XGBClassifier(**default_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        optimal_threshold, _ = self.find_optimal_threshold(y_val, y_pred_proba)
        metrics = self.evaluate_model(y_val, y_pred_proba, optimal_threshold)
        metrics['optimal_threshold'] = optimal_threshold
        
        return model, metrics
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        params: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Train Random Forest model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            params: Optional custom parameters
            
        Returns:
            Tuple of (model, metrics)
        """
        default_params = {
            'n_estimators': 200,
            'max_depth': 12,
            'min_samples_split': 100,
            'min_samples_leaf': 50,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': self.random_seed,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        optimal_threshold, _ = self.find_optimal_threshold(y_val, y_pred_proba)
        metrics = self.evaluate_model(y_val, y_pred_proba, optimal_threshold)
        metrics['optimal_threshold'] = optimal_threshold
        
        return model, metrics
    
    def train_decision_tree(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        params: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Train Decision Tree baseline model.
        
        Using hyperparameters from teacher's notebook analysis:
        - max_depth: 8
        - min_samples_leaf: 20
        - ccp_alpha: 0.001
        - class_weight: balanced
        """
        default_params = {
            'max_depth': 8,
            'min_samples_leaf': 20,
            'ccp_alpha': 0.001,
            'class_weight': 'balanced',
            'random_state': self.random_seed
        }
        
        if params:
            default_params.update(params)
            
        model = DecisionTreeClassifier(**default_params)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        optimal_threshold, _ = self.find_optimal_threshold(y_val, y_pred_proba)
        metrics = self.evaluate_model(y_val, y_pred_proba, optimal_threshold)
        metrics['optimal_threshold'] = optimal_threshold
        
        return model, metrics

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Train and compare all available models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Dictionary of model results
        """
        logger.info("Training all models...")
        
        # Calculate class weights
        class_weights, scale_pos_weight = self.calculate_class_weights(y_train)
        
        results = {}
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM...")
            try:
                model, metrics = self.train_lightgbm(
                    X_train, y_train, X_val, y_val, scale_pos_weight
                )
                self.models['LightGBM'] = model
                results['LightGBM'] = metrics
                logger.info(f"LightGBM PR-AUC: {metrics['pr_auc']:.4f}")
                logger.info(f"LightGBM ROC-AUC: {metrics['roc_auc']:.4f}")
            except Exception as e:
                logger.error(f"LightGBM training failed: {e}")
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost...")
            try:
                model, metrics = self.train_xgboost(
                    X_train, y_train, X_val, y_val, scale_pos_weight
                )
                self.models['XGBoost'] = model
                results['XGBoost'] = metrics
                logger.info(f"XGBoost PR-AUC: {metrics['pr_auc']:.4f}")
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        try:
            model, metrics = self.train_random_forest(
                X_train, y_train, X_val, y_val
            )
            self.models['RandomForest'] = model
            results['RandomForest'] = metrics
            logger.info(f"Random Forest PR-AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"Random Forest ROC-AUC: {metrics['roc_auc']:.4f}")
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")

        # Train Decision Tree Baseline
        logger.info("Training Decision Tree Baseline...")
        try:
            model, metrics = self.train_decision_tree(
                X_train, y_train, X_val, y_val
            )
            self.models['DecisionTree'] = model
            results['DecisionTree'] = metrics
            logger.info(f"Decision Tree PR-AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"Decision Tree ROC-AUC: {metrics['roc_auc']:.4f}")
        except Exception as e:
            logger.error(f"Decision Tree training failed: {e}")
        
        self.results = results
        
        # Select best model based on PR-AUC
        if results:
            self.best_model_name = max(results.keys(), key=lambda k: results[k]['pr_auc'])
            self.best_model = self.models[self.best_model_name]
            self.best_threshold = results[self.best_model_name]['optimal_threshold']
            
            logger.info(f"Best model: {self.best_model_name} "
                       f"(PR-AUC: {results[self.best_model_name]['pr_auc']:.4f})")
        
        return results
    
    def save_model(
        self,
        path: str,
        feature_cols: List[str],
        include_all_models: bool = False
    ) -> None:
        """
        Save trained model(s) and artifacts.
        
        Args:
            path: Path to save model
            feature_cols: List of feature column names
            include_all_models: Whether to save all models or just best
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'optimal_threshold': self.best_threshold,
            'feature_cols': feature_cols,
            'results': self.results,
            'random_seed': self.random_seed,
            'training_date': datetime.now().isoformat()
        }
        
        if include_all_models:
            artifacts['all_models'] = self.models
        
        with open(path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Model saved to {path}")
    
    def log_to_mlflow(
        self,
        model_name: str,
        params: Dict,
        metrics: Dict,
        model: Any
    ) -> str:
        """
        Log model training run to MLflow.
        
        Args:
            model_name: Name of the model
            params: Model parameters
            metrics: Evaluation metrics
            model: Trained model object
            
        Returns:
            MLflow run ID
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping logging")
            return ""
        
        with mlflow.start_run(run_name=model_name) as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log model
            if 'LightGBM' in model_name and LIGHTGBM_AVAILABLE:
                mlflow.lightgbm.log_model(model, 'model')
            elif 'XGBoost' in model_name and XGBOOST_AVAILABLE:
                mlflow.xgboost.log_model(model, 'model')
            else:
                mlflow.sklearn.log_model(model, 'model')
            
            return run.info.run_id


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save model')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--experiment-name', type=str, default='fraud_detection',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    train_df = pd.read_parquet(Path(args.data_path) / 'train_processed.parquet')
    
    # Load feature artifacts
    with open(Path(args.data_path).parent / 'features' / 'feature_artifacts.pkl', 'rb') as f:
        feature_artifacts = pickle.load(f)
    
    feature_cols = feature_artifacts['feature_cols']
    
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
    
    logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(
        random_seed=args.random_seed,
        experiment_name=args.experiment_name
    )
    
    # Train models
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Save best model
    trainer.save_model(args.output_path, available_features)
    
    # Print results
    print("\nModel Comparison Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Threshold: {metrics['optimal_threshold']:.4f}")
    
    print(f"\nBest Model: {trainer.best_model_name}")


if __name__ == '__main__':
    main()
