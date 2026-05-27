"""
Comprehensive model evaluation metrics utilities.

Functions for calculating and displaying various ML evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, log_loss
)


def evaluate_classification(y_true, y_pred, y_pred_proba=None, average='weighted'):
    """
    Comprehensive classification evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        average: Averaging method for multi-class

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Add AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                # Multi-class
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba,
                                                   multi_class='ovr', average=average)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except Exception as e:
            print(f"Could not calculate AUC: {e}")

    return metrics


def evaluate_regression(y_true, y_pred):
    """
    Comprehensive regression evaluation.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
    }

    return metrics


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def print_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Print formatted classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    """
    metrics = evaluate_classification(y_true, y_pred, y_pred_proba)

    print("=" * 60)
    print("  CLASSIFICATION METRICS")
    print("=" * 60)

    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")

    print("\n" + "=" * 60)
    print("  DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred))

    print("=" * 60)
    print("  CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print()


def print_regression_metrics(y_true, y_pred):
    """
    Print formatted regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
    """
    metrics = evaluate_regression(y_true, y_pred)

    print("=" * 60)
    print("  REGRESSION METRICS")
    print("=" * 60)

    print(f"{'MAE':15s}: {metrics['mae']:.4f}")
    print(f"{'MSE':15s}: {metrics['mse']:.4f}")
    print(f"{'RMSE':15s}: {metrics['rmse']:.4f}")
    print(f"{'R² Score':15s}: {metrics['r2']:.4f}")
    print(f"{'MAPE':15s}: {metrics['mape']:.2f}%")

    print("=" * 60)


def compare_models(models_dict, X_test, y_test, metric='accuracy', problem_type='classification'):
    """
    Compare multiple models on the same test set.

    Args:
        models_dict: Dictionary of {model_name: trained_model}
        X_test: Test features
        y_test: Test labels/values
        metric: Metric to compare ('accuracy', 'f1_score', 'rmse', etc.)
        problem_type: 'classification' or 'regression'

    Returns:
        DataFrame with comparison results
    """
    results = []

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)

        if problem_type == 'classification':
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            metrics = evaluate_classification(y_test, y_pred, y_pred_proba)
        else:
            metrics = evaluate_regression(y_test, y_pred)

        metrics['model'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index('model')

    # Sort by specified metric
    if metric in df.columns:
        ascending = metric in ['mae', 'mse', 'rmse', 'log_loss']
        df = df.sort_values(metric, ascending=ascending)

    return df


def calculate_confusion_matrix_metrics(cm):
    """
    Calculate detailed metrics from confusion matrix.

    Args:
        cm: Confusion matrix (2x2 for binary classification)

    Returns:
        Dictionary of metrics
    """
    if cm.shape != (2, 2):
        raise ValueError("Only binary classification supported")

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
    }

    return metrics


def cross_validation_summary(cv_scores, metric_name='Score'):
    """
    Summarize cross-validation results.

    Args:
        cv_scores: Array of cross-validation scores
        metric_name: Name of the metric

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'mean': np.mean(cv_scores),
        'std': np.std(cv_scores),
        'min': np.min(cv_scores),
        'max': np.max(cv_scores),
        'median': np.median(cv_scores),
    }

    print(f"\n{metric_name} - Cross-Validation Summary:")
    print(f"  Mean:   {summary['mean']:.4f}")
    print(f"  Std:    {summary['std']:.4f}")
    print(f"  Min:    {summary['min']:.4f}")
    print(f"  Max:    {summary['max']:.4f}")
    print(f"  Median: {summary['median']:.4f}")
    print(f"  Individual scores: {cv_scores}")

    return summary


if __name__ == "__main__":
    print("Evaluation metrics utilities loaded successfully!")
