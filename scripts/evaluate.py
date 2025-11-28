#!/usr/bin/env python3
"""
Evaluation Script

Evaluate model on test data.

Usage:
    python scripts/evaluate.py --test-data data/test.csv --model-name fqdn_classifier
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl

from src.evaluation.evaluator import Evaluator, evaluate_model
from src.inference.predictor import create_predictor
from src.utils.helpers import timer
from src.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FQDN classification model")

    parser.add_argument("--test-data", type=str, required=True, help="Path to test CSV data")
    parser.add_argument("--model-path", type=str, default="models/trained", help="Path to model directory")
    parser.add_argument("--model-name", type=str, default="fqdn_classifier", help="Model name")
    parser.add_argument("--model-version", type=str, help="Model version")
    parser.add_argument("--output-report", type=str, default="reports/evaluation.json", help="Output report path")
    parser.add_argument("--compute-shap", action="store_true", help="Compute SHAP values")
    parser.add_argument("--error-analysis", action="store_true", help="Perform error analysis")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


@timer("Total evaluation time")
def main():
    args = parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(level=log_level)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("FQDN Classification - Model Evaluation")
    logger.info("=" * 60)

    # Load predictor and extract components
    logger.info(f"Loading model: {args.model_name}")
    predictor = create_predictor(
        model_path=args.model_path,
        model_name=args.model_name,
        version=args.model_version,
    )

    # Load test data
    logger.info(f"Loading test data: {args.test_data}")
    df = pl.read_csv(args.test_data)
    logger.info(f"Loaded {len(df)} test records")

    # Preprocess and extract features
    if predictor.preprocessor:
        df = predictor.preprocessor.transform(df)

    X, feature_names = predictor.feature_pipeline.transform(df)

    # Get true labels
    target_col = "appid"
    y_true = df[target_col].to_numpy()

    # Encode labels
    if predictor.class_mapping:
        y_true = np.array(predictor.class_mapping.transform(y_true.tolist()))

    # Create evaluator
    evaluator = Evaluator(
        predictor.model,
        feature_names=feature_names,
    )

    # Evaluate
    logger.info("Running evaluation...")
    results = evaluator.evaluate(X, y_true, compute_shap=args.compute_shap)

    # Error analysis
    if args.error_analysis:
        logger.info("Performing error analysis...")
        error_analysis = evaluator.get_error_analysis(X, y_true)
        logger.info(f"Total errors: {error_analysis['total_errors']}")
        logger.info(f"Error rate: {error_analysis['error_rate']:.4f}")
        logger.info("Top confusion pairs:")
        for pair in error_analysis["confusion_pairs"][:5]:
            logger.info(f"  {pair['true']} -> {pair['predicted']}: {pair['count']}")

    # Save report
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(results, output_path)

    logger.info("=" * 60)
    logger.info(f"Evaluation report saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
