"""
Evaluation Module
Functions for model evaluation, statistical tests, and comparison.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values

    Returns:
    --------
    dict : Dictionary containing RMSE, MAE, R², MAPE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def kolmogorov_smirnov_test(real_data: pd.DataFrame,
                            synthetic_data: pd.DataFrame,
                            columns: list = None,
                            alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform Kolmogorov-Smirnov test to compare distributions.

    Parameters:
    -----------
    real_data : DataFrame
        Real dataset
    synthetic_data : DataFrame
        Synthetic dataset
    columns : list, optional
        Columns to test (if None, uses all numeric columns)
    alpha : float
        Significance level

    Returns:
    --------
    DataFrame : KS test results with statistics and p-values
    """
    if columns is None:
        columns = real_data.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for col in columns:
        try:
            statistic, p_value = stats.ks_2samp(real_data[col], synthetic_data[col])
            passed = p_value > alpha
            results.append({
                'Feature': col,
                'KS_Statistic': statistic,
                'P_Value': p_value,
                'Passed': passed
            })
        except Exception as e:
            print(f"Error testing {col}: {e}")

    return pd.DataFrame(results)


def paired_ttest(baseline_errors: np.ndarray,
                augmented_errors: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test to compare model errors.

    Parameters:
    -----------
    baseline_errors : array-like
        Absolute errors from baseline model
    augmented_errors : array-like
        Absolute errors from augmented model

    Returns:
    --------
    tuple : (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(baseline_errors, augmented_errors)
    return t_stat, p_value


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Parameters:
    -----------
    group1 : array-like
        First group of values
    group2 : array-like
        Second group of values

    Returns:
    --------
    float : Cohen's d effect size
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
    return mean_diff / pooled_std


def compare_models(y_true: np.ndarray,
                  y_pred_baseline: np.ndarray,
                  y_pred_augmented: np.ndarray) -> Dict:
    """
    Comprehensive comparison of baseline vs augmented models.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred_baseline : array-like
        Baseline model predictions
    y_pred_augmented : array-like
        Augmented model predictions

    Returns:
    --------
    dict : Comparison results including metrics and statistical tests
    """
    # Calculate metrics for both models
    baseline_metrics = calculate_regression_metrics(y_true, y_pred_baseline)
    augmented_metrics = calculate_regression_metrics(y_true, y_pred_augmented)

    # Calculate errors
    baseline_errors = np.abs(y_true - y_pred_baseline)
    augmented_errors = np.abs(y_true - y_pred_augmented)

    # Statistical tests
    t_stat, p_value = paired_ttest(baseline_errors, augmented_errors)
    effect_size = cohens_d(baseline_errors, augmented_errors)

    # Calculate improvements
    improvements = {}
    for metric in ['RMSE', 'MAE', 'MAPE']:
        baseline_val = baseline_metrics[metric]
        augmented_val = augmented_metrics[metric]
        improvements[metric] = ((baseline_val - augmented_val) / baseline_val) * 100

    # R2 improvement is different (higher is better)
    improvements['R2'] = ((augmented_metrics['R2'] - baseline_metrics['R2']) /
                         baseline_metrics['R2']) * 100

    return {
        'baseline_metrics': baseline_metrics,
        'augmented_metrics': augmented_metrics,
        'improvements': improvements,
        'statistical_tests': {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': effect_size,
            'significant': p_value < 0.05
        }
    }


def generate_comparison_table(baseline_metrics: Dict[str, float],
                              augmented_metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Generate comparison table for baseline vs augmented models.

    Parameters:
    -----------
    baseline_metrics : dict
        Baseline model metrics
    augmented_metrics : dict
        Augmented model metrics

    Returns:
    --------
    DataFrame : Comparison table
    """
    comparison = []

    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        augmented_val = augmented_metrics[metric]

        # Calculate improvement (lower is better for RMSE, MAE, MAPE; higher for R2)
        if metric in ['RMSE', 'MAE', 'MAPE']:
            improvement = ((baseline_val - augmented_val) / baseline_val) * 100
        else:  # R2
            improvement = ((augmented_val - baseline_val) / baseline_val) * 100

        comparison.append({
            'Metric': metric,
            'Baseline': baseline_val,
            'Augmented': augmented_val,
            'Improvement (%)': improvement
        })

    return pd.DataFrame(comparison)


def evaluate_by_subgroup(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        groups: np.ndarray) -> pd.DataFrame:
    """
    Evaluate model performance by subgroups (e.g., aircraft type).

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    groups : array-like
        Group labels for each sample

    Returns:
    --------
    DataFrame : Metrics by subgroup
    """
    results = []
    unique_groups = np.unique(groups)

    for group in unique_groups:
        mask = groups == group
        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]

        if len(group_y_true) > 0:
            metrics = calculate_regression_metrics(group_y_true, group_y_pred)
            metrics['Group'] = group
            metrics['Count'] = len(group_y_true)
            results.append(metrics)

    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_regression_metrics()")
    print("  - kolmogorov_smirnov_test()")
    print("  - paired_ttest()")
    print("  - cohens_d()")
    print("  - compare_models()")
    print("  - generate_comparison_table()")
    print("  - evaluate_by_subgroup()")
