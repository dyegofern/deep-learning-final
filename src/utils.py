"""
Utility functions for GAN-Based Carbon Emissions Prediction Project
CSCA 5642 - Final Project
University of Colorado Boulder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


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


def plot_predictions_vs_actuals(y_true: np.ndarray, y_pred: np.ndarray,
                                title: str = 'Predictions vs Actuals',
                                save_path: str = None) -> None:
    """
    Create scatter plot of predictions vs actual values.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate metrics
    metrics = calculate_regression_metrics(y_true, y_pred)

    # Add metrics text box
    textstr = f"RMSE: {metrics['RMSE']:.2f}\nMAE: {metrics['MAE']:.2f}\nR²: {metrics['R2']:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_xlabel('Actual CO₂ Emissions (kg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted CO₂ Emissions (kg)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = 'Residual Analysis',
                   save_path: str = None) -> None:
    """
    Create residual plots for model diagnostics.

    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Residual scatter plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, color='coral', edgecolors='black', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted CO₂ Emissions (kg)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residuals (kg)', fontsize=12, fontweight='bold')
    axes[0].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (kg)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def compare_distributions(real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                         columns: List[str] = None,
                         save_path: str = None) -> None:
    """
    Compare distributions of real vs synthetic data.

    Parameters:
    -----------
    real_data : DataFrame
        Real dataset
    synthetic_data : DataFrame
        Synthetic dataset
    columns : list, optional
        Columns to compare (if None, uses all numeric columns)
    save_path : str, optional
        Path to save the plot
    """
    if columns is None:
        columns = real_data.select_dtypes(include=[np.number]).columns.tolist()

    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() if len(columns) > 1 else [axes]

    for idx, col in enumerate(columns):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Plot histograms
        ax.hist(real_data[col], bins=30, alpha=0.6, label='Real', color='steelblue', edgecolor='black')
        ax.hist(synthetic_data[col], bins=30, alpha=0.6, label='Synthetic', color='coral', edgecolor='black')

        ax.set_xlabel(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Real vs Synthetic Data Distribution Comparison', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def kolmogorov_smirnov_test(real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                            columns: List[str] = None,
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


def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                           top_n: int = 20,
                           title: str = 'Feature Importance',
                           save_path: str = None) -> None:
    """
    Plot feature importance as horizontal bar chart.

    Parameters:
    -----------
    importances : array-like
        Feature importance values
    feature_names : list
        Feature names
    top_n : int
        Number of top features to display
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Create plot
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

    plt.barh(range(len(top_features)), top_importances, color=colors, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_training_history(history: Dict[str, List[float]],
                         metrics: List[str] = ['loss'],
                         title: str = 'Training History',
                         save_path: str = None) -> None:
    """
    Plot training history (loss curves, etc.).

    Parameters:
    -----------
    history : dict
        Training history dictionary
    metrics : list
        Metrics to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        if metric in history:
            ax.plot(history[metric], linewidth=2, label=f'Training {metric}')

        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], linewidth=2, label=f'Validation {metric}', linestyle='--')

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def save_metrics_report(metrics: Dict[str, float],
                       model_name: str,
                       save_path: str) -> None:
    """
    Save metrics report to text file.

    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    model_name : str
        Name of the model
    save_path : str
        Path to save the report
    """
    with open(save_path, 'w') as f:
        f.write(f"Model Evaluation Report: {model_name}\n")
        f.write("=" * 60 + "\n\n")

        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Metrics report saved to: {save_path}")


def print_section_header(title: str, char: str = '=') -> None:
    """
    Print formatted section header.

    Parameters:
    -----------
    title : str
        Section title
    char : str
        Character to use for header line
    """
    width = 70
    print('\n' + char * width)
    print(title.center(width))
    print(char * width + '\n')


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

        # Calculate improvement (lower is better for RMSE, MAE, MAPE; higher is better for R2)
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


if __name__ == '__main__':
    print("Utility functions loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_regression_metrics()")
    print("  - plot_predictions_vs_actuals()")
    print("  - plot_residuals()")
    print("  - compare_distributions()")
    print("  - kolmogorov_smirnov_test()")
    print("  - plot_feature_importance()")
    print("  - plot_training_history()")
    print("  - save_metrics_report()")
    print("  - print_section_header()")
    print("  - cohens_d()")
    print("  - generate_comparison_table()")
