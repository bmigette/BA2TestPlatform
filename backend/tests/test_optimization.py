"""
Test genetic optimization with minimal parameters.

Tests the full optimization pipeline with small values to ensure it works end-to-end.

Note: On macOS with MPS (Apple Silicon), small datasets may cause MPS tensor bugs.
Use larger datasets (500+ rows) or force CPU training on Mac.
"""

import pytest
import sys
import os
import platform

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.job_handler import handle_training_job

# Check if running on Mac with MPS - these tests may be flaky
IS_MAC = platform.system() == "Darwin"


# Minimal payload for quick optimization test
MINIMAL_OPTIMIZATION_PAYLOAD = {
    "job_type": "classification",
    "dataset_ids": [1],  # Will be updated with actual dataset ID
    "selected_models": ["lstm"],  # Single model
    "parameter_ranges": {
        # Only 2-3 values per parameter to minimize combinations
        "layersMin": 1,
        "layersMax": 2,
        "layersStep": 1,  # 2 values: 1, 2
        "layerSizeMin": 32,
        "layerSizeMax": 64,
        "layerSizeStep": 32,  # 2 values: 32, 64
        "learningRateMin": 0.001,
        "learningRateMax": 0.001,
        "learningRateStep": 0.001,  # 1 value: 0.001
        "dropoutMin": 0.0,
        "dropoutMax": 0.0,
        "dropoutStep": 0.1,  # 1 value: 0.0
        "seqLen": 16,  # Small sequence length
        "normalizationBuffer": 35
    },
    "prediction_targets": [
        {
            "type": "directional",
            "category": "binary_classification",
            "config": {
                "horizon": 1
            }
        }
    ],
    "prediction_horizon": 1,
    "prediction_modes": ["shift"],
    "train_test_split": 80,
    "cross_validation": None,
    "genetic_config": {
        "populationSize": 4,  # Very small population
        "generations": 2,  # Only 2 generations
        "elitismPercent": 25.0,  # 1 elite individual
        "crossoverProb": 0.7,
        "mutationProb": 0.2,
        "earlyStoppingGenerations": 5,
        "trainingEpochs": 2  # Minimal epochs
    },
    "metrics_config": {
        "optimizeMetric": "f1_score",
        "classificationMetric": "f1_score",
        "regressionMetric": "rmse",
        "lossFunction": "focal_loss"
    },
    "training_date_range": None
}


class TestOptimization:
    """Test genetic optimization end-to-end."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Check if we can connect to the database and get a dataset."""
        try:
            from app.models.database import get_db
            from app.models.dataset import Dataset
            db = next(get_db())
            dataset = db.query(Dataset).first()
            if dataset is None:
                pytest.skip("No datasets in database - run with a populated database")
            self.dataset_id = dataset.id
            self.dataset_rows = dataset.rows_count
            # Need enough data for train/test split + sequences
            if self.dataset_rows < 200:
                pytest.skip(f"Dataset too small ({self.dataset_rows} rows) - need at least 200")
        except Exception as e:
            pytest.skip(f"Database not available: {e}")

    def _update_payload_dataset(self, payload: dict) -> dict:
        """Update payload to use actual dataset ID."""
        payload = payload.copy()
        payload['dataset_ids'] = [self.dataset_id]
        return payload

    def test_minimal_optimization(self):
        """
        Test optimization with minimal parameters.

        This tests the full pipeline:
        - Data preparation with normalization
        - Genetic algorithm initialization
        - Fitness function evaluation
        - Model training (2 epochs per individual)
        - 2 generations of optimization
        - Results collection
        """
        payload = self._update_payload_dataset(MINIMAL_OPTIMIZATION_PAYLOAD)

        # Run actual optimization (not dry_run)
        result = handle_training_job("test-minimal-opt", payload, dry_run=False)

        # Check result structure
        assert result['status'] in ['completed', 'partial'], f"Job failed: {result.get('error', result)}"

        # Should have model results
        assert 'model_results' in result, f"Missing model_results: {result}"
        assert len(result['model_results']) > 0, "No model results"

        # Check first model result
        model_result = result['model_results'][0]
        assert model_result['status'] == 'completed', f"Model failed: {model_result.get('error', model_result)}"
        assert 'best_params' in model_result, "Missing best_params"
        assert 'best_fitness' in model_result, "Missing best_fitness"
        assert 'generations_run' in model_result, "Missing generations_run"

        # Should have run at least 1 generation
        assert model_result['generations_run'] >= 1, f"Only ran {model_result['generations_run']} generations"

        # Best fitness should be a valid number
        assert isinstance(model_result['best_fitness'], (int, float)), "best_fitness not a number"
        assert model_result['best_fitness'] >= 0, "Negative fitness"

        # Best params should have required keys
        best_params = model_result['best_params']
        assert 'n_layers' in best_params, "Missing n_layers in best_params"
        assert 'hidden_size' in best_params, "Missing hidden_size in best_params"

        print(f"Optimization completed successfully:")
        print(f"  Generations run: {model_result['generations_run']}")
        print(f"  Best fitness: {model_result['best_fitness']:.4f}")
        print(f"  Best params: {model_result['best_params']}")

    def test_optimization_with_multistep(self):
        """Test optimization with multistep prediction mode."""
        payload = self._update_payload_dataset(MINIMAL_OPTIMIZATION_PAYLOAD)
        payload['prediction_modes'] = ['multistep']
        payload['prediction_horizon'] = 3
        # Use cross_entropy for multistep (focal_loss not supported)
        payload['metrics_config']['lossFunction'] = 'cross_entropy'

        result = handle_training_job("test-multistep-opt", payload, dry_run=False)

        assert result['status'] in ['completed', 'partial'], f"Job failed: {result.get('error', result)}"
        assert 'model_results' in result

        model_result = result['model_results'][0]
        assert model_result['status'] == 'completed', f"Model failed: {model_result.get('error', model_result)}"

        print(f"Multistep optimization completed:")
        print(f"  Best fitness: {model_result['best_fitness']:.4f}")


def run_quick_optimization_test():
    """Run a quick optimization test without pytest."""
    print("=" * 60)
    print("Running quick optimization test...")
    print("=" * 60)

    try:
        from app.models.database import get_db
        from app.models.dataset import Dataset
        db = next(get_db())
        dataset = db.query(Dataset).first()
        if dataset is None:
            print("ERROR: No datasets in database")
            return False
        dataset_id = dataset.id
        print(f"Using dataset ID: {dataset_id} ({dataset.rows_count} rows)")
    except Exception as e:
        print(f"ERROR: Cannot access database: {e}")
        return False

    payload = MINIMAL_OPTIMIZATION_PAYLOAD.copy()
    payload['dataset_ids'] = [dataset_id]

    print("\nRunning minimal optimization (2 generations, population 4)...")
    result = handle_training_job("test-quick-opt", payload, dry_run=False)

    if result['status'] not in ['completed', 'partial']:
        print(f"FAILED: {result.get('error', result)}")
        return False

    if 'model_results' not in result or len(result['model_results']) == 0:
        print(f"FAILED: No model results")
        return False

    model_result = result['model_results'][0]
    if model_result['status'] != 'completed':
        print(f"FAILED: Model status = {model_result['status']}")
        print(f"Error: {model_result.get('error')}")
        return False

    print("\n" + "=" * 60)
    print("Optimization test PASSED!")
    print("=" * 60)
    print(f"Generations run: {model_result['generations_run']}")
    print(f"Best fitness: {model_result['best_fitness']:.4f}")
    print(f"Best params: {model_result['best_params']}")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test genetic optimization")
    parser.add_argument("--quick", action="store_true", help="Run quick test without pytest")
    args = parser.parse_args()

    if args.quick:
        success = run_quick_optimization_test()
        exit(0 if success else 1)
    else:
        # Run with pytest
        pytest.main([__file__, "-v"])
