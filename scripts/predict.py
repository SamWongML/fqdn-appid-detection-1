#!/usr/bin/env python3
"""
Prediction Script

Batch prediction for orphan FQDNs.

Usage:
    python scripts/predict.py --input data/orphans.csv --output predictions.csv
    python scripts/predict.py --input data/orphans.csv --model-name fqdn_classifier
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from src.inference.predictor import Predictor, create_predictor
from src.utils.helpers import timer
from src.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict appId for orphan FQDNs")

    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV file with FQDNs")
    parser.add_argument("--output", "-o", type=str, default="predictions.csv", help="Output CSV file")
    parser.add_argument("--model-path", type=str, default="models/trained", help="Path to model directory")
    parser.add_argument("--model-name", type=str, default="fqdn_classifier", help="Model name")
    parser.add_argument("--model-version", type=str, help="Model version (default: latest)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    parser.add_argument("--include-explanation", action="store_true", help="Include feature explanations")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


@timer("Total prediction time")
def main():
    args = parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    setup_logger(level=log_level)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("FQDN Classification - Batch Prediction")
    logger.info("=" * 60)

    # Load predictor
    logger.info(f"Loading model: {args.model_name}")
    predictor = create_predictor(
        model_path=args.model_path,
        model_name=args.model_name,
        version=args.model_version,
    )
    predictor.confidence_threshold = args.confidence_threshold
    predictor.top_k = args.top_k

    # Load input data
    logger.info(f"Loading input data: {args.input}")
    df = pl.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records")

    # Process in batches
    all_results = []
    n_batches = (len(df) + args.batch_size - 1) // args.batch_size

    for i in range(n_batches):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(df))
        batch_df = df.slice(start_idx, end_idx - start_idx)

        logger.info(f"Processing batch {i+1}/{n_batches} ({len(batch_df)} records)")

        results = predictor.predict_batch(batch_df, include_explanation=args.include_explanation)
        all_results.extend(results.predictions)

    # Create output DataFrame
    output_records = []
    for pred in all_results:
        record = {
            "fqdn": pred.fqdn,
            "predicted_appid": pred.predicted_appid,
            "confidence": pred.confidence,
            "is_uncertain": pred.is_uncertain,
        }
        for j, (appid, prob) in enumerate(pred.top_k_predictions):
            record[f"top_{j+1}_appid"] = appid
            record[f"top_{j+1}_prob"] = prob
        output_records.append(record)

    output_df = pl.DataFrame(output_records)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_csv(output_path)

    # Summary statistics
    uncertain_count = sum(1 for p in all_results if p.is_uncertain)
    avg_confidence = sum(p.confidence for p in all_results) / len(all_results)

    logger.info("=" * 60)
    logger.info("Prediction Summary")
    logger.info("=" * 60)
    logger.info(f"Total predictions: {len(all_results)}")
    logger.info(f"Uncertain predictions: {uncertain_count} ({100*uncertain_count/len(all_results):.1f}%)")
    logger.info(f"Average confidence: {avg_confidence:.4f}")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
