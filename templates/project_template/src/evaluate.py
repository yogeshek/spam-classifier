"""
Model evaluation script template.

This script provides comprehensive model evaluation including:
- Performance metrics
- Visualizations
- Cross-validation
- Feature importance

Usage:
    python evaluate.py --model models/model.pkl --data data/test.csv
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score


def load_model(model_path):
    """Load trained model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_data(data_path, target_column):
    """Load test data."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def evaluate_classification(model, X, y, output_dir):
    """
    Comprehensive evaluation for classification models.
    """
    print("\n" + "=" * 70)
    print("  CLASSIFICATION METRICS")
    print("=" * 70)

    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')

    # ROC Curve (for binary classification)
    if y_pred_proba is not None and len(np.unique(y)) == 2:
        plot_roc_curve(y, y_pred_proba, output_dir / 'roc_curve.png')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_regression(model, X, y, output_dir):
    """
    Comprehensive evaluation for regression models.
    """
    print("\n" + "=" * 70)
    print("  REGRESSION METRICS")
    print("=" * 70)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\nMSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # Residual plot
    plot_residuals(y, y_pred, output_dir / 'residuals.png')

    # Actual vs Predicted
    plot_actual_vs_predicted(y, y_pred, output_dir / 'actual_vs_predicted.png')

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(y_true, y_scores, output_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ROC curve to {output_path}")


def plot_residuals(y_true, y_pred, output_path):
    """Plot residuals for regression."""
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved residual plot to {output_path}")


def plot_actual_vs_predicted(y_true, y_pred, output_path):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved actual vs predicted plot to {output_path}")


def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance if available."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved feature importance plot to {output_path}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate ML model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--target', type=str, default='target',
                        help='Name of target column')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for plots and metrics')
    parser.add_argument('--type', type=str, choices=['classification', 'regression'],
                        default='classification', help='Model type')

    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL EVALUATION")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model = load_model(args.model)
    X, y = load_data(args.data, args.target)

    # Evaluate based on model type
    if args.type == 'classification':
        metrics = evaluate_classification(model, X, y, output_dir)
    else:
        metrics = evaluate_regression(model, X, y, output_dir)

    # Feature importance
    plot_feature_importance(model, X.columns, output_dir / 'feature_importance.png')

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / 'metrics.csv', index=False)

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE!")
    print(f"  Results saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
