"""
Fraud Detection MLOps - Source Package

This package contains production-ready modules for the
IEEE-CIS Fraud Detection pipeline.

Modules:
    - data_processing: Data loading, validation, and quality checks
    - feature_engineering: Feature transformation pipeline
    - train: Model training with MLflow integration
    - predict: Inference pipeline with monitoring
"""

from .data_processing import DataProcessor, run_data_quality_checks
from .feature_engineering import FeatureEngineer
from .train import FraudDetectionTrainer
from .predict import FraudDetectionPredictor, create_mock_scoring_function

__version__ = '1.0.0'
__author__ = 'Fraud Detection MLOps Team'

__all__ = [
    'DataProcessor',
    'FeatureEngineer',
    'FraudDetectionTrainer',
    'FraudDetectionPredictor',
    'run_data_quality_checks',
    'create_mock_scoring_function'
]
