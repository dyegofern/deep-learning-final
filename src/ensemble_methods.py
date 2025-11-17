"""
Ensemble Methods for Combining Models
CSCA 5642 - Final Project Enhancement
University of Colorado Boulder

This module implements ensemble strategies:
1. Simple Averaging
2. Weighted Ensemble
3. Stacking Ensemble
4. Dynamic Weighting based on confidence
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Optional, Tuple
import pickle


class SimpleAveragingEnsemble:
    """
    Simple averaging ensemble that combines predictions from multiple models.
    """

    def __init__(self, models: List):
        """
        Initialize simple averaging ensemble.

        Parameters:
        -----------
        models : list
            List of trained models
        """
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions by averaging all model predictions.

        Parameters:
        -----------
        X : array
            Input features

        Returns:
        --------
        array : Averaged predictions
        """
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

    def get_individual_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from each individual model."""
        return np.array([model.predict(X) for model in self.models])


class WeightedEnsemble:
    """
    Weighted ensemble that combines predictions with learned weights.
    """

    def __init__(self, models: List, weights: Optional[np.ndarray] = None):
        """
        Initialize weighted ensemble.

        Parameters:
        -----------
        models : list
            List of trained models
        weights : array, optional
            Weights for each model (must sum to 1)
        """
        self.models = models
        self.n_models = len(models)

        if weights is None:
            # Equal weights by default
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            assert len(weights) == self.n_models, "Weights must match number of models"
            assert np.isclose(np.sum(weights), 1.0), "Weights must sum to 1"
            self.weights = np.array(weights)

    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Optimize weights using validation data to minimize MSE.

        Parameters:
        -----------
        X_val : array
            Validation features
        y_val : array
            Validation targets
        """
        from scipy.optimize import minimize

        # Get predictions from all models
        predictions = np.array([model.predict(X_val) for model in self.models])

        def mse_loss(weights):
            """Compute MSE for given weights."""
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.sum(weights[:, None] * predictions, axis=0)
            return np.mean((y_val - ensemble_pred) ** 2)

        # Initialize with equal weights
        initial_weights = np.ones(self.n_models) / self.n_models

        # Optimize with constraints (weights must be positive and sum to 1)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_models)]

        result = minimize(
            mse_loss,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.weights = result.x
        print(f"Optimized weights: {self.weights}")

        return self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted average.

        Parameters:
        -----------
        X : array
            Input features

        Returns:
        --------
        array : Weighted predictions
        """
        predictions = np.array([model.predict(X) for model in self.models])
        return np.sum(self.weights[:, None] * predictions, axis=0)


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    """

    def __init__(self,
                 base_models: List,
                 meta_model=None,
                 use_original_features: bool = True):
        """
        Initialize stacking ensemble.

        Parameters:
        -----------
        base_models : list
            List of base models
        meta_model : model, optional
            Meta-learner model (default: Ridge regression)
        use_original_features : bool
            Whether to include original features for meta-learner
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.use_original_features = use_original_features
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Fit meta-learner on base model predictions.

        Parameters:
        -----------
        X_train : array
            Training features (for generating meta-features)
        y_train : array
            Training targets
        X_val : array, optional
            Validation features for meta-learner training
        y_val : array, optional
            Validation targets for meta-learner training
        """
        # If no validation set provided, use training set (may overfit)
        if X_val is None:
            X_val = X_train
            y_val = y_train

        # Generate meta-features from base model predictions
        meta_features = self._generate_meta_features(X_val)

        # Optionally include original features
        if self.use_original_features:
            meta_features = np.hstack([X_val, meta_features])

        # Train meta-learner
        self.meta_model.fit(meta_features, y_val)
        self.is_fitted = True

        print(f"Stacking ensemble trained with {len(self.base_models)} base models")

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        predictions = np.array([model.predict(X) for model in self.base_models])
        return predictions.T  # Shape: (n_samples, n_models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.

        Parameters:
        -----------
        X : array
            Input features

        Returns:
        --------
        array : Stacked predictions
        """
        assert self.is_fitted, "Stacking ensemble must be fitted first"

        # Generate meta-features
        meta_features = self._generate_meta_features(X)

        # Optionally include original features
        if self.use_original_features:
            meta_features = np.hstack([X, meta_features])

        # Meta-learner prediction
        return self.meta_model.predict(meta_features)


class DynamicWeightedEnsemble:
    """
    Dynamic weighted ensemble that adjusts weights based on prediction confidence.
    """

    def __init__(self, models: List, confidence_metric: str = 'variance'):
        """
        Initialize dynamic weighted ensemble.

        Parameters:
        -----------
        models : list
            List of trained models (must support prediction variance)
        confidence_metric : str
            Metric for confidence ('variance', 'std', or 'agreement')
        """
        self.models = models
        self.confidence_metric = confidence_metric

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with dynamic weights based on confidence.

        Parameters:
        -----------
        X : array
            Input features

        Returns:
        --------
        array : Dynamically weighted predictions
        """
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])

        # Compute confidence-based weights
        if self.confidence_metric == 'variance':
            # Lower variance = higher confidence = higher weight
            variances = np.var(predictions, axis=0, keepdims=True)
            # Inverse variance weighting
            weights = 1.0 / (variances + 1e-8)
            weights = weights / np.sum(weights, axis=0, keepdims=True)

            # Use uniform weights where variance is too high (unreliable)
            high_variance_mask = variances[0] > np.percentile(variances, 75)
            weights[:, high_variance_mask] = 1.0 / len(self.models)

        elif self.confidence_metric == 'agreement':
            # Higher agreement = higher confidence
            mean_pred = np.mean(predictions, axis=0, keepdims=True)
            agreements = 1.0 / (np.abs(predictions - mean_pred) + 1e-8)
            weights = agreements / np.sum(agreements, axis=0, keepdims=True)

        else:
            # Default: equal weights
            weights = np.ones_like(predictions) / len(self.models)

        # Weighted average
        weighted_pred = np.sum(weights * predictions, axis=0)

        return weighted_pred


def build_ensemble(models: List,
                   ensemble_type: str = 'averaging',
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> object:
    """
    Build an ensemble from a list of models.

    Parameters:
    -----------
    models : list
        List of trained models
    ensemble_type : str
        Type of ensemble ('averaging', 'weighted', 'stacking', 'dynamic')
    X_val : array, optional
        Validation features (for weighted/stacking)
    y_val : array, optional
        Validation targets (for weighted/stacking)

    Returns:
    --------
    object : Ensemble model
    """
    if ensemble_type == 'averaging':
        ensemble = SimpleAveragingEnsemble(models)
        print("Built simple averaging ensemble")

    elif ensemble_type == 'weighted':
        ensemble = WeightedEnsemble(models)
        if X_val is not None and y_val is not None:
            ensemble.optimize_weights(X_val, y_val)
            print("Built weighted ensemble with optimized weights")
        else:
            print("Built weighted ensemble with equal weights")

    elif ensemble_type == 'stacking':
        ensemble = StackingEnsemble(models)
        if X_val is not None and y_val is not None:
            ensemble.fit(X_val, y_val)
            print("Built stacking ensemble with meta-learner")
        else:
            raise ValueError("Stacking ensemble requires validation data")

    elif ensemble_type == 'dynamic':
        ensemble = DynamicWeightedEnsemble(models)
        print("Built dynamic weighted ensemble")

    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    return ensemble


def evaluate_ensemble(ensemble,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      return_predictions: bool = False) -> Dict:
    """
    Evaluate ensemble performance.

    Parameters:
    -----------
    ensemble : Ensemble
        Trained ensemble model
    X_test : array
        Test features
    y_test : array
        Test targets
    return_predictions : bool
        Whether to return predictions

    Returns:
    --------
    dict : Evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Make predictions
    y_pred = ensemble.predict(X_test)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    if return_predictions:
        results['predictions'] = y_pred

    return results


if __name__ == '__main__':
    print("Ensemble methods loaded successfully!")
    print("\nAvailable ensemble types:")
    print("  - SimpleAveragingEnsemble")
    print("  - WeightedEnsemble")
    print("  - StackingEnsemble")
    print("  - DynamicWeightedEnsemble")
    print("\nHelper function:")
    print("  - build_ensemble()")
    print("  - evaluate_ensemble()")
