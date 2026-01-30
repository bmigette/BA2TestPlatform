"""
F1 Metric Investigation Test Script

Investigates why models are predicting all 1s (F1=0.051, recall=1.0, precision=0.026).
Tests multiple configurations to identify the root cause:

1. Different prediction targets (ZigZag vs Directional vs Price-Based)
2. Different model types (LSTM, InceptionTime)
3. Different loss functions (focal_loss, cross_entropy, weighted_cross_entropy)
4. Different thresholds (0.3, 0.5, 0.7)

Outputs detailed JSON report for analysis.

Usage:
    cd backend
    ./venv/bin/python tests/test_f1_investigation.py

Results saved to: tests/f1_investigation_results.json
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.job_handler import handle_training_job
from app.models.database import get_db
from app.models.dataset import Dataset

# Configuration
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "f1_investigation_results.json")

# Test configurations
TEST_CONFIGS = {
    "epochs": 50,
    "population_size": 6,  # Small for faster testing
    "generations": 3,  # Few generations to test convergence
    "seq_len": 24,
    "train_test_split": 80,
}

# Model types to test
MODEL_TYPES = ["lstm", "inception"]

# Loss functions to test
LOSS_FUNCTIONS = ["focal_loss", "cross_entropy", "weighted_cross_entropy"]

# Thresholds to test
THRESHOLDS = [0.3, 0.5, 0.7]

# Prediction targets to test
PREDICTION_TARGETS = {
    "zigzag_1pct": {
        "name": "ZigZag 1%",
        "target": {
            "type": "trend_reversal",
            "category": "binary_classification",
            "config": {
                "indicator": "zigzag",
                "direction": "bullish",
                "threshold": 1.0,
                "indicatorParams": {"threshold": 1.0}
            }
        }
    },
    "zigzag_2pct": {
        "name": "ZigZag 2%",
        "target": {
            "type": "trend_reversal",
            "category": "binary_classification",
            "config": {
                "indicator": "zigzag",
                "direction": "bullish",
                "threshold": 2.0,
                "indicatorParams": {"threshold": 2.0}
            }
        }
    },
    "directional_1bar": {
        "name": "Directional 1-bar",
        "target": {
            "type": "directional",
            "category": "binary_classification",
            "config": {
                "horizon": 1
            }
        }
    },
    "directional_3bar": {
        "name": "Directional 3-bar",
        "target": {
            "type": "directional",
            "category": "binary_classification",
            "config": {
                "horizon": 3
            }
        }
    },
    "price_based_5pct": {
        "name": "Price-Based 5% profit",
        "target": {
            "type": "price_based",
            "category": "binary_classification",
            "config": {
                "direction": "up",
                "profitTarget": 5.0,
                "maxDrawdown": 3.0,
                "timeWindow": 10
            }
        }
    },
}


def get_dataset_info() -> Dict[str, Any]:
    """Get first available dataset info."""
    try:
        db = next(get_db())
        dataset = db.query(Dataset).first()
        if dataset is None:
            return None
        return {
            "id": dataset.id,
            "name": dataset.name,
            "rows": dataset.rows_count,
            "ticker": dataset.ticker if hasattr(dataset, 'ticker') else "unknown"
        }
    except Exception as e:
        print(f"Error getting dataset: {e}")
        return None


def create_test_payload(
    dataset_id: int,
    model_type: str,
    loss_function: str,
    threshold: float,
    target_config: Dict
) -> Dict[str, Any]:
    """Create a test payload for a specific configuration."""
    return {
        "job_type": "classification",
        "dataset_ids": [dataset_id],
        "selected_models": [model_type],
        "parameter_ranges": {
            "layersMin": 1,
            "layersMax": 2,
            "layersStep": 1,
            "layerSizeMin": 64,
            "layerSizeMax": 128,
            "layerSizeStep": 64,
            "learningRateMin": 0.001,
            "learningRateMax": 0.001,
            "learningRateStep": 0.001,
            "dropoutMin": 0.1,
            "dropoutMax": 0.2,
            "dropoutStep": 0.1,
            "seqLen": TEST_CONFIGS["seq_len"],
            "normalizationBuffer": 35
        },
        "prediction_targets": [target_config["target"]],
        "prediction_horizon": target_config["target"].get("config", {}).get("horizon", 1),
        "prediction_modes": ["shift"],
        "train_test_split": TEST_CONFIGS["train_test_split"],
        "cross_validation": None,
        "genetic_config": {
            "populationSize": TEST_CONFIGS["population_size"],
            "generations": TEST_CONFIGS["generations"],
            "elitismPercent": 20.0,
            "crossoverProb": 0.7,
            "mutationProb": 0.2,
            "earlyStoppingGenerations": 5,
            "trainingEpochs": TEST_CONFIGS["epochs"]
        },
        "metrics_config": {
            "optimizeMetric": "f1_score",
            "classificationMetric": "f1_score",
            "regressionMetric": "rmse",
            "lossFunction": loss_function,
            "lossFunctions": [loss_function],
            "optimizeLossFunction": False,
            "thresholdMin": threshold,
            "thresholdMax": threshold,  # Fixed threshold for this test
            "thresholdStep": 0.1
        },
        "training_date_range": None
    }


def analyze_metrics(metrics: Dict) -> Dict[str, Any]:
    """Analyze metrics to detect prediction pattern issues."""
    analysis = {
        "all_ones": False,
        "all_zeros": False,
        "balanced": False,
        "issue": None
    }

    if not metrics:
        analysis["issue"] = "no_metrics"
        return analysis

    recall = metrics.get("recall", 0)
    precision = metrics.get("precision", 0)
    tn = metrics.get("true_negatives", 0)
    fp = metrics.get("false_positives", 0)
    fn = metrics.get("false_negatives", 0)
    tp = metrics.get("true_positives", 0)

    total_predictions = tp + fp + fn + tn
    if total_predictions == 0:
        analysis["issue"] = "no_predictions"
        return analysis

    # Check for all-ones prediction (recall=1, tn=0)
    if recall == 1.0 and tn == 0:
        analysis["all_ones"] = True
        analysis["issue"] = "predicting_all_positives"
    # Check for all-zeros prediction (recall=0, fp=0)
    elif recall == 0 and fp == 0:
        analysis["all_zeros"] = True
        analysis["issue"] = "predicting_all_negatives"
    else:
        analysis["balanced"] = True

    # Calculate actual positive ratio
    actual_positives = tp + fn
    actual_negatives = tn + fp
    if actual_positives + actual_negatives > 0:
        analysis["positive_ratio"] = actual_positives / (actual_positives + actual_negatives)

    # Calculate predicted positive ratio
    predicted_positives = tp + fp
    if total_predictions > 0:
        analysis["predicted_positive_ratio"] = predicted_positives / total_predictions

    return analysis


def run_single_test(
    test_id: str,
    dataset_id: int,
    model_type: str,
    loss_function: str,
    threshold: float,
    target_key: str,
    target_config: Dict
) -> Dict[str, Any]:
    """Run a single test configuration and return results."""

    print(f"\n{'='*60}")
    print(f"Test: {test_id}")
    print(f"  Model: {model_type}")
    print(f"  Loss: {loss_function}")
    print(f"  Threshold: {threshold}")
    print(f"  Target: {target_config['name']}")
    print(f"{'='*60}")

    result = {
        "test_id": test_id,
        "config": {
            "model_type": model_type,
            "loss_function": loss_function,
            "threshold": threshold,
            "target_key": target_key,
            "target_name": target_config["name"],
            "epochs": TEST_CONFIGS["epochs"],
            "population_size": TEST_CONFIGS["population_size"],
            "generations": TEST_CONFIGS["generations"]
        },
        "status": None,
        "best_fitness": None,
        "metrics": None,
        "analysis": None,
        "generations_run": None,
        "error": None,
        "duration_seconds": None,
        "timestamp": datetime.now().isoformat()
    }

    payload = create_test_payload(
        dataset_id=dataset_id,
        model_type=model_type,
        loss_function=loss_function,
        threshold=threshold,
        target_config=target_config
    )

    start_time = time.time()

    try:
        job_result = handle_training_job(test_id, payload, dry_run=False)

        result["duration_seconds"] = round(time.time() - start_time, 2)
        result["status"] = job_result.get("status", "unknown")

        if job_result.get("status") in ["completed", "partial"]:
            results = job_result.get("results", [])
            if results and len(results) > 0:
                model_result = results[0]
                result["best_fitness"] = model_result.get("best_fitness")
                result["generations_run"] = model_result.get("generations_run")
                result["metrics"] = model_result.get("metrics", {})
                result["analysis"] = analyze_metrics(result["metrics"])

                # Log key findings
                if result["analysis"]:
                    if result["analysis"].get("all_ones"):
                        print(f"  ⚠️  ISSUE: Predicting all positives!")
                    elif result["analysis"].get("all_zeros"):
                        print(f"  ⚠️  ISSUE: Predicting all negatives!")
                    else:
                        print(f"  ✓ Balanced predictions")

                print(f"  F1: {result['metrics'].get('f1_score', 'N/A'):.4f}" if result['metrics'].get('f1_score') else "  F1: N/A")
                print(f"  Recall: {result['metrics'].get('recall', 'N/A')}")
                print(f"  Precision: {result['metrics'].get('precision', 'N/A')}")
        else:
            result["error"] = job_result.get("error", "Unknown error")
            print(f"  ✗ Failed: {result['error']}")

    except Exception as e:
        result["duration_seconds"] = round(time.time() - start_time, 2)
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ✗ Exception: {e}")

    return result


def run_all_tests(dataset_id: int) -> List[Dict[str, Any]]:
    """Run all test configurations."""
    all_results = []
    test_count = 0
    total_tests = len(MODEL_TYPES) * len(LOSS_FUNCTIONS) * len(THRESHOLDS) * len(PREDICTION_TARGETS)

    print(f"\nTotal tests to run: {total_tests}")
    print(f"Estimated time: {total_tests * 5} - {total_tests * 15} minutes")
    print(f"(depending on model convergence and early stopping)\n")

    for target_key, target_config in PREDICTION_TARGETS.items():
        for model_type in MODEL_TYPES:
            for loss_function in LOSS_FUNCTIONS:
                for threshold in THRESHOLDS:
                    test_count += 1
                    test_id = f"f1_test_{test_count:03d}_{model_type}_{loss_function}_{threshold}_{target_key}"

                    print(f"\n[{test_count}/{total_tests}] Running test...")

                    result = run_single_test(
                        test_id=test_id,
                        dataset_id=dataset_id,
                        model_type=model_type,
                        loss_function=loss_function,
                        threshold=threshold,
                        target_key=target_key,
                        target_config=target_config
                    )

                    all_results.append(result)

                    # Save intermediate results after each test
                    save_results(all_results, dataset_id)

    return all_results


def save_results(results: List[Dict[str, Any]], dataset_id: int):
    """Save results to JSON file."""
    output = {
        "investigation": "F1 Metric Issue - All Ones Prediction",
        "created_at": datetime.now().isoformat(),
        "dataset_id": dataset_id,
        "test_config": TEST_CONFIGS,
        "model_types_tested": MODEL_TYPES,
        "loss_functions_tested": LOSS_FUNCTIONS,
        "thresholds_tested": THRESHOLDS,
        "targets_tested": list(PREDICTION_TARGETS.keys()),
        "total_tests": len(results),
        "summary": generate_summary(results),
        "results": results
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_FILE}")


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from all results."""
    summary = {
        "total_tests": len(results),
        "completed": 0,
        "failed": 0,
        "all_ones_predictions": 0,
        "all_zeros_predictions": 0,
        "balanced_predictions": 0,
        "best_f1_overall": 0,
        "best_config": None,
        "by_target": {},
        "by_model": {},
        "by_loss_function": {},
        "by_threshold": {}
    }

    for result in results:
        if result["status"] in ["completed", "partial"]:
            summary["completed"] += 1

            analysis = result.get("analysis", {})
            if analysis.get("all_ones"):
                summary["all_ones_predictions"] += 1
            elif analysis.get("all_zeros"):
                summary["all_zeros_predictions"] += 1
            elif analysis.get("balanced"):
                summary["balanced_predictions"] += 1

            f1 = result.get("metrics", {}).get("f1_score", 0) or 0
            if f1 > summary["best_f1_overall"]:
                summary["best_f1_overall"] = f1
                summary["best_config"] = result["config"]
        else:
            summary["failed"] += 1

        # Group by target
        target = result["config"]["target_key"]
        if target not in summary["by_target"]:
            summary["by_target"][target] = {"count": 0, "all_ones": 0, "balanced": 0, "best_f1": 0}
        summary["by_target"][target]["count"] += 1
        if result.get("analysis", {}).get("all_ones"):
            summary["by_target"][target]["all_ones"] += 1
        if result.get("analysis", {}).get("balanced"):
            summary["by_target"][target]["balanced"] += 1
        f1 = result.get("metrics", {}).get("f1_score", 0) or 0
        if f1 > summary["by_target"][target]["best_f1"]:
            summary["by_target"][target]["best_f1"] = f1

        # Group by model
        model = result["config"]["model_type"]
        if model not in summary["by_model"]:
            summary["by_model"][model] = {"count": 0, "all_ones": 0, "balanced": 0, "best_f1": 0}
        summary["by_model"][model]["count"] += 1
        if result.get("analysis", {}).get("all_ones"):
            summary["by_model"][model]["all_ones"] += 1
        if result.get("analysis", {}).get("balanced"):
            summary["by_model"][model]["balanced"] += 1
        if f1 > summary["by_model"][model]["best_f1"]:
            summary["by_model"][model]["best_f1"] = f1

        # Group by loss function
        loss = result["config"]["loss_function"]
        if loss not in summary["by_loss_function"]:
            summary["by_loss_function"][loss] = {"count": 0, "all_ones": 0, "balanced": 0, "best_f1": 0}
        summary["by_loss_function"][loss]["count"] += 1
        if result.get("analysis", {}).get("all_ones"):
            summary["by_loss_function"][loss]["all_ones"] += 1
        if result.get("analysis", {}).get("balanced"):
            summary["by_loss_function"][loss]["balanced"] += 1
        if f1 > summary["by_loss_function"][loss]["best_f1"]:
            summary["by_loss_function"][loss]["best_f1"] = f1

        # Group by threshold
        thresh = result["config"]["threshold"]
        thresh_key = str(thresh)
        if thresh_key not in summary["by_threshold"]:
            summary["by_threshold"][thresh_key] = {"count": 0, "all_ones": 0, "balanced": 0, "best_f1": 0}
        summary["by_threshold"][thresh_key]["count"] += 1
        if result.get("analysis", {}).get("all_ones"):
            summary["by_threshold"][thresh_key]["all_ones"] += 1
        if result.get("analysis", {}).get("balanced"):
            summary["by_threshold"][thresh_key]["balanced"] += 1
        if f1 > summary["by_threshold"][thresh_key]["best_f1"]:
            summary["by_threshold"][thresh_key]["best_f1"] = f1

    return summary


def print_final_summary(results: List[Dict[str, Any]]):
    """Print a human-readable summary."""
    summary = generate_summary(results)

    print("\n" + "="*70)
    print("INVESTIGATION SUMMARY")
    print("="*70)

    print(f"\nTotal tests: {summary['total_tests']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")

    print(f"\nPrediction Pattern Analysis:")
    print(f"  All-ones (recall=1, tn=0): {summary['all_ones_predictions']}")
    print(f"  All-zeros (recall=0, fp=0): {summary['all_zeros_predictions']}")
    print(f"  Balanced predictions: {summary['balanced_predictions']}")

    print(f"\nBest F1 Score: {summary['best_f1_overall']:.4f}")
    if summary['best_config']:
        print(f"  Config: {summary['best_config']}")

    print("\nBy Prediction Target:")
    for target, stats in summary["by_target"].items():
        print(f"  {target}: {stats['balanced']}/{stats['count']} balanced, best F1={stats['best_f1']:.4f}")

    print("\nBy Model Type:")
    for model, stats in summary["by_model"].items():
        print(f"  {model}: {stats['balanced']}/{stats['count']} balanced, best F1={stats['best_f1']:.4f}")

    print("\nBy Loss Function:")
    for loss, stats in summary["by_loss_function"].items():
        print(f"  {loss}: {stats['balanced']}/{stats['count']} balanced, best F1={stats['best_f1']:.4f}")

    print("\nBy Threshold:")
    for thresh, stats in summary["by_threshold"].items():
        print(f"  {thresh}: {stats['balanced']}/{stats['count']} balanced, best F1={stats['best_f1']:.4f}")

    print("\n" + "="*70)
    print(f"Full results saved to: {RESULTS_FILE}")
    print("="*70)


def main():
    """Main entry point."""
    print("="*70)
    print("F1 METRIC INVESTIGATION")
    print("Investigating why models predict all 1s (F1≈0.051)")
    print("="*70)

    # Get dataset
    dataset_info = get_dataset_info()
    if not dataset_info:
        print("ERROR: No dataset available. Please ensure database has at least one dataset.")
        sys.exit(1)

    print(f"\nUsing dataset: {dataset_info['name']} (ID: {dataset_info['id']}, {dataset_info['rows']} rows)")

    if dataset_info['rows'] < 500:
        print(f"WARNING: Dataset only has {dataset_info['rows']} rows. May cause issues with train/test split.")

    print(f"\nTest Parameters:")
    print(f"  Epochs: {TEST_CONFIGS['epochs']}")
    print(f"  Population: {TEST_CONFIGS['population_size']}")
    print(f"  Generations: {TEST_CONFIGS['generations']}")
    print(f"  Sequence Length: {TEST_CONFIGS['seq_len']}")
    print(f"\nModels: {MODEL_TYPES}")
    print(f"Loss Functions: {LOSS_FUNCTIONS}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Targets: {list(PREDICTION_TARGETS.keys())}")

    # Run all tests
    results = run_all_tests(dataset_info['id'])

    # Print final summary
    print_final_summary(results)


if __name__ == "__main__":
    main()
