"""
Improved Training Functions for CTGAN
CSCA 5642 - Final Project Enhancement

Key improvements:
1. Longer training with early stopping
2. Learning rate scheduling
3. Better monitoring of training progress
4. Validation during training
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
from scipy import stats as scipy_stats


def train_improved_ctgan(ctgan,
                         real_data: np.ndarray,
                         epochs: int = 300,
                         batch_size: int = 500,
                         n_critic: int = 5,
                         verbose: bool = True,
                         early_stopping_patience: int = 50,
                         validation_interval: int = 10) -> Dict:
    """
    Train Improved CTGAN with monitoring and early stopping.

    Parameters:
    -----------
    ctgan : ImprovedCTGAN
        CTGAN model to train
    real_data : array
        Transformed real data
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size
    n_critic : int
        Discriminator updates per generator update
    verbose : bool
        Print progress
    early_stopping_patience : int
        Epochs to wait for improvement before stopping
    validation_interval : int
        Epochs between validation checks

    Returns:
    --------
    dict : Training history
    """
    n_samples = len(real_data)

    history = {
        'g_loss': [],
        'd_loss': [],
        'w_distance': [],
        'gp': [],
        'best_epoch': 0
    }

    best_w_distance = float('inf')
    patience_counter = 0

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(real_data)
    dataset = dataset.shuffle(buffer_size=n_samples)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    iterator = tqdm(range(epochs), desc='Training Improved CTGAN') if verbose else range(epochs)

    for epoch in iterator:
        epoch_g_loss = []
        epoch_d_loss = []
        epoch_w_dist = []
        epoch_gp = []

        for real_batch in dataset:
            # Train discriminator n_critic times
            for _ in range(n_critic):
                d_loss, w_dist, gp = ctgan.train_discriminator_step(real_batch)

            epoch_d_loss.append(float(d_loss))
            epoch_w_dist.append(float(w_dist))
            epoch_gp.append(float(gp))

            # Train generator once
            g_loss = ctgan.train_generator_step(batch_size)
            epoch_g_loss.append(float(g_loss))

        # Record averages
        avg_g_loss = np.mean(epoch_g_loss)
        avg_d_loss = np.mean(epoch_d_loss)
        avg_w_dist = np.mean(epoch_w_dist)
        avg_gp = np.mean(epoch_gp)

        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['w_distance'].append(avg_w_dist)
        history['gp'].append(avg_gp)

        if verbose:
            iterator.set_postfix({
                'G_loss': f'{avg_g_loss:.4f}',
                'D_loss': f'{avg_d_loss:.4f}',
                'W_dist': f'{avg_w_dist:.4f}',
                'GP': f'{avg_gp:.4f}'
            })

        # Early stopping based on Wasserstein distance
        if epoch % validation_interval == 0:
            if abs(avg_w_dist) < abs(best_w_distance):
                best_w_distance = avg_w_dist
                history['best_epoch'] = epoch
                patience_counter = 0
            else:
                patience_counter += validation_interval

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best Wasserstein distance: {best_w_distance:.4f} at epoch {history['best_epoch']}")
                break

    return history


def validate_synthetic_quality(real_data: np.ndarray,
                                synthetic_data: np.ndarray,
                                column_names: List[str],
                                continuous_cols: List[int],
                                binary_cols: List[int],
                                onehot_groups: List[List[int]]) -> Dict:
    """
    Comprehensive validation of synthetic data quality.

    Parameters:
    -----------
    real_data : array
        Real data
    synthetic_data : array
        Synthetic data
    column_names : list
        Column names
    continuous_cols : list
        Indices of continuous columns
    binary_cols : list
        Indices of binary columns
    onehot_groups : list
        Groups of one-hot encoded columns

    Returns:
    --------
    dict : Validation metrics
    """
    from scipy.stats import ks_2samp, wasserstein_distance
    from sklearn.metrics import mean_squared_error

    results = {
        'continuous': [],
        'binary': [],
        'onehot': [],
        'overall_quality': 0.0
    }

    # Validate continuous columns
    for col_idx in continuous_cols:
        real_col = real_data[:, col_idx]
        syn_col = synthetic_data[:, col_idx]

        ks_stat, ks_p = ks_2samp(real_col, syn_col)
        w_dist = wasserstein_distance(real_col, syn_col)

        results['continuous'].append({
            'column': column_names[col_idx],
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'wasserstein_distance': w_dist,
            'passed': ks_p > 0.05
        })

    # Validate binary columns
    for col_idx in binary_cols:
        real_col = real_data[:, col_idx]
        syn_col = synthetic_data[:, col_idx]

        # Check distribution of 0s and 1s
        real_mean = np.mean(real_col)
        syn_mean = np.mean(syn_col)
        diff = abs(real_mean - syn_mean)

        results['binary'].append({
            'column': column_names[col_idx],
            'real_mean': real_mean,
            'synthetic_mean': syn_mean,
            'difference': diff,
            'passed': diff < 0.1  # Within 10%
        })

    # Validate one-hot groups
    for group_idx, group in enumerate(onehot_groups):
        real_group = real_data[:, group]
        syn_group = synthetic_data[:, group]

        # Check if sums to 1 (valid one-hot)
        real_sums = np.sum(real_group, axis=1)
        syn_sums = np.sum(syn_group, axis=1)

        real_valid = np.mean(np.isclose(real_sums, 1.0))
        syn_valid = np.mean(np.isclose(syn_sums, 1.0))

        # Check distribution across categories
        real_dist = np.mean(real_group, axis=0)
        syn_dist = np.mean(syn_group, axis=0)

        chi2_stat = np.sum((real_dist - syn_dist) ** 2 / (real_dist + 1e-8))

        results['onehot'].append({
            'group': f'Group_{group_idx}',
            'real_validity': real_valid,
            'synthetic_validity': syn_valid,
            'chi2_statistic': chi2_stat,
            'passed': syn_valid > 0.95 and chi2_stat < 0.1
        })

    # Compute overall quality score
    continuous_passed = sum(r['passed'] for r in results['continuous']) / max(len(results['continuous']), 1)
    binary_passed = sum(r['passed'] for r in results['binary']) / max(len(results['binary']), 1)
    onehot_passed = sum(r['passed'] for r in results['onehot']) / max(len(results['onehot']), 1)

    results['overall_quality'] = (continuous_passed + binary_passed + onehot_passed) / 3

    # Summary
    results['summary'] = {
        'continuous_pass_rate': continuous_passed * 100,
        'binary_pass_rate': binary_passed * 100,
        'onehot_pass_rate': onehot_passed * 100,
        'overall_pass_rate': results['overall_quality'] * 100
    }

    return results


def print_validation_report(validation_results: Dict):
    """
    Print detailed validation report.

    Parameters:
    -----------
    validation_results : dict
        Results from validate_synthetic_quality
    """
    print("\n" + "="*80)
    print("SYNTHETIC DATA QUALITY VALIDATION REPORT")
    print("="*80)

    # Continuous features
    print("\n1. CONTINUOUS FEATURES:")
    print("-" * 80)
    for result in validation_results['continuous']:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {result['column']:<30} {status}")
        print(f"    KS Statistic: {result['ks_statistic']:.6f}, p-value: {result['ks_p_value']:.6f}")
        print(f"    Wasserstein Distance: {result['wasserstein_distance']:.6f}")

    # Binary features
    if validation_results['binary']:
        print("\n2. BINARY FEATURES:")
        print("-" * 80)
        for result in validation_results['binary']:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {result['column']:<30} {status}")
            print(f"    Real mean: {result['real_mean']:.4f}, Synthetic mean: {result['synthetic_mean']:.4f}")
            print(f"    Difference: {result['difference']:.4f}")

    # One-hot groups
    if validation_results['onehot']:
        print("\n3. ONE-HOT ENCODED GROUPS:")
        print("-" * 80)
        for result in validation_results['onehot']:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {result['group']:<30} {status}")
            print(f"    Validity: Real={result['real_validity']:.2%}, Synthetic={result['synthetic_validity']:.2%}")
            print(f"    Chi-square: {result['chi2_statistic']:.6f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    summary = validation_results['summary']
    print(f"  Continuous features pass rate: {summary['continuous_pass_rate']:.1f}%")
    print(f"  Binary features pass rate: {summary['binary_pass_rate']:.1f}%")
    print(f"  One-hot groups pass rate: {summary['onehot_pass_rate']:.1f}%")
    print(f"  Overall quality score: {summary['overall_pass_rate']:.1f}%")
    print("="*80)


if __name__ == '__main__':
    print("Improved training functions loaded successfully!")
