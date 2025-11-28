"""
FQDN Orphan Detection ML System

A machine learning system for classifying orphan FQDNs (Fully Qualified Domain Names)
to their corresponding application IDs.
"""

__version__ = "1.0.0"
__author__ = "FQDN ML Team"

from src.config.settings import get_settings
from src.training.trainer import Trainer, create_trainer
from src.inference.predictor import Predictor, create_predictor
from src.evaluation.evaluator import Evaluator, evaluate_model

__all__ = [
    "get_settings",
    "Trainer",
    "create_trainer",
    "Predictor",
    "create_predictor",
    "Evaluator",
    "evaluate_model",
]
