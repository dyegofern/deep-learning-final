"""
Training Module
Functions for training CTGAN and baseline models.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple
from tqdm import tqdm


def train_ctgan(ctgan,
                real_data: np.ndarray,
                condition: np.ndarray = None,
                epochs: int = 1000,
                batch_size: int = 128,
                n_critic: int = 5,
                verbose: bool = True) -> Dict[str, List[float]]:
    """
    Train CTGAN model with Wasserstein loss and gradient penalty.

    Parameters:
    -----------
    ctgan : CTGAN
        CTGAN model instance
    real_data : array-like
        Real training data
    condition : array-like, optional
        Conditional input for CTGAN
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    n_critic : int
        Number of discriminator updates per generator update
    verbose : bool
        Whether to display progress bar

    Returns:
    --------
    dict : Training history with loss values
    """
    n_samples = len(real_data)
    history = {
        'g_loss': [],
        'd_loss': [],
        'w_distance': [],
        'gp': []
    }

    # Progress bar
    iterator = tqdm(range(epochs), desc='Training CTGAN') if verbose else range(epochs)

    for epoch in iterator:
        epoch_g_loss = []
        epoch_d_loss = []
        epoch_w_dist = []
        epoch_gp = []

        # Number of batches
        n_batches = n_samples // batch_size

        for batch_idx in range(n_batches):
            # Get batch of real data
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            real_batch = real_data[start_idx:end_idx]

            batch_condition = None
            if condition is not None:
                batch_condition = condition[start_idx:end_idx]

            # Train discriminator n_critic times
            for _ in range(n_critic):
                d_loss, w_dist, gp = ctgan.train_discriminator_step(
                    real_batch, batch_condition
                )
                epoch_d_loss.append(float(d_loss))
                epoch_w_dist.append(float(w_dist))
                epoch_gp.append(float(gp))

            # Train generator once
            g_loss = ctgan.train_generator_step(batch_size, batch_condition)
            epoch_g_loss.append(float(g_loss))

        # Record average losses for epoch
        history['g_loss'].append(np.mean(epoch_g_loss))
        history['d_loss'].append(np.mean(epoch_d_loss))
        history['w_distance'].append(np.mean(epoch_w_dist))
        history['gp'].append(np.mean(epoch_gp))

        # Update progress bar
        if verbose and (epoch + 1) % 10 == 0:
            iterator.set_postfix({
                'G_loss': f"{history['g_loss'][-1]:.4f}",
                'D_loss': f"{history['d_loss'][-1]:.4f}",
                'W_dist': f"{history['w_distance'][-1]:.4f}"
            })

    return history


def generate_synthetic_data(ctgan,
                            n_samples: int,
                            condition: np.ndarray = None) -> np.ndarray:
    """
    Generate synthetic data using trained CTGAN.

    Parameters:
    -----------
    ctgan : CTGAN
        Trained CTGAN model
    n_samples : int
        Number of synthetic samples to generate
    condition : array-like, optional
        Conditional input for generation

    Returns:
    --------
    array : Synthetic data samples
    """
    synthetic_data = ctgan.generate_samples(n_samples, condition)
    return synthetic_data


def train_random_forest(X_train: np.ndarray,
                       y_train: np.ndarray,
                       n_estimators: int = 100,
                       max_depth: int = 20,
                       min_samples_split: int = 2,
                       min_samples_leaf: int = 1,
                       min_weight_fraction_leaf: float = 0.0,
                       random_state: int = 42,
                       verbose: bool = True) -> RandomForestRegressor:
    """
    Train Random Forest regression model.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
    random_state : int
        Random seed
    verbose : bool
        Whether to display progress

    Returns:
    --------
    RandomForestRegressor : Trained model
    """
    if verbose:
        print(f"Training Random Forest with {n_estimators} trees...")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )

    model.fit(X_train, y_train)

    if verbose:
        print("✓ Training complete!")

    return model


def train_baseline_model(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         model_type: str = 'rf',
                         min_samples_split: int = 2,
                         **kwargs) -> object:
    """
    Train baseline model (Random Forest or other).

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    model_type : str
        Type of model ('rf' for Random Forest)
    **kwargs : dict
        Additional parameters for model

    Returns:
    --------
    object : Trained model
    """
    if model_type == 'rf':
        return train_random_forest(X_train.values, y_train.values,  min_samples_split=min_samples_split, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_augmented_model(X_real: pd.DataFrame,
                         y_real: pd.Series,
                         X_synthetic: pd.DataFrame,
                         y_synthetic: pd.Series,
                         model_type: str = 'rf',
                         **kwargs) -> object:
    """
    Train model on augmented dataset (real + synthetic data).

    Parameters:
    -----------
    X_real : DataFrame
        Real training features
    y_real : Series
        Real training target
    X_synthetic : DataFrame
        Synthetic training features
    y_synthetic : Series
        Synthetic training target
    model_type : str
        Type of model
    **kwargs : dict
        Additional model parameters

    Returns:
    --------
    object : Trained model
    """
    # Combine real and synthetic data
    X_augmented = pd.concat([X_real, X_synthetic], axis=0, ignore_index=True)
    y_augmented = pd.concat([y_real, y_synthetic], axis=0, ignore_index=True)

    print(f"Training on augmented dataset:")
    print(f"  Real samples: {len(X_real)}")
    print(f"  Synthetic samples: {len(X_synthetic)}")
    print(f"  Total samples: {len(X_augmented)}")

    # Train model
    model = train_baseline_model(X_augmented, y_augmented, model_type, **kwargs)

    return model


if __name__ == '__main__':
    print("Training module loaded successfully!")
    print("\nAvailable functions:")
    print("  - train_ctgan()")
    print("  - generate_synthetic_data()")
    print("  - train_random_forest()")
    print("  - train_baseline_model()")
    print("  - train_augmented_model()")
