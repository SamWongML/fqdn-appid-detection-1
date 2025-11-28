#!/usr/bin/env python3
"""
Training Script

Main entry point for training the FQDN classification model.

Usage:
    python scripts/train.py --labeled-data data/labeled.csv --model-type ensemble
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, timer
from src.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FQDN classification model")

    parser.add_argument("--labeled-data", type=str, help="Path to labeled CSV data")
    parser.add_argument("--unlabeled-data", type=str, help="Path to unlabeled CSV data")
    parser.add_argument("--source", type=str, choices=["csv", "postgres", "auto"], default="auto")
    parser.add_argument(
        "--model-type", type=str, choices=["xgboost", "lightgbm", "catboost", "ensemble"], default="ensemble"
    )
    parser.add_argument("--model-name", type=str, default="fqdn_classifier")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


@timer("Total training time")
def main():
    args = parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(level=log_level)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("FQDN Classification Model Training")
    logger.info("=" * 60)

    set_seed(args.seed)

    trainer = Trainer(experiment_name=args.experiment_name, use_wandb=not args.no_wandb)

    try:
        # Load data
        labeled_df, unlabeled_df = trainer.load_data(
            source=args.source,
            labeled_path=args.labeled_data,
            unlabeled_path=args.unlabeled_data,
        )

        # Preprocess
        labeled_df = trainer.preprocess(labeled_df, fit=True)

        # Engineer features
        X, y, feature_names = trainer.engineer_features(labeled_df, fit=True)

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=args.seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.176, stratify=y_train, random_state=args.seed
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Cross-validation (optional)
        if args.cross_validate:
            trainer.cross_validate(X_train, y_train, n_splits=args.cv_folds, model_type=args.model_type)

        # Hyperparameter optimization (optional)
        if args.optimize:
            best_params = trainer.optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=args.n_trials,
                model_type=args.model_type if args.model_type != "ensemble" else "xgboost",
            )

        # Train final model
        model = trainer.train(X_train, y_train, X_val, y_val, model_type=args.model_type)

        # Evaluate on test set
        from src.evaluation.evaluator import Evaluator

        evaluator = Evaluator(model, feature_names)
        test_results = evaluator.evaluate(X_test, y_test)

        # Save report
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        evaluator.save_report(test_results, reports_dir / f"{args.model_name}_evaluation.json")

        # Save model
        model_path = trainer.save_model(model_name=args.model_name)
        logger.info(f"Model saved to: {model_path}")

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Test Accuracy: {test_results.accuracy:.4f}")
        logger.info(f"Test F1 (macro): {test_results.f1_macro:.4f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        trainer.finish()


if __name__ == "__main__":
    main()
