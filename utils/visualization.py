"""
Shared visualization utilities for ML projects.

Common plotting functions for exploratory data analysis and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_distribution(df, column, bins=30, title=None):
    """
    Plot distribution of a numerical column.

    Args:
        df: DataFrame
        column: Column name
        bins: Number of bins for histogram
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    axes[0].hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {column}' if title is None else title)
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot(df[column].dropna(), vert=True)
    axes[1].set_ylabel(column)
    axes[1].set_title(f'Box Plot of {column}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_categorical_distribution(df, column, top_n=10):
    """
    Plot distribution of a categorical column.

    Args:
        df: DataFrame
        column: Column name
        top_n: Number of top categories to show
    """
    value_counts = df[column].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {column} (Top {top_n})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_correlation_matrix(df, figsize=(12, 10), annot=True):
    """
    Plot correlation matrix heatmap.

    Args:
        df: DataFrame with numerical columns
        figsize: Figure size
        annot: Whether to annotate cells with correlation values
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['float64', 'int64'])

    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f',
                cmap='coolwarm', center=0, square=True, ax=ax,
                linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix')
    plt.tight_layout()

    return fig


def plot_scatter_with_regression(df, x_column, y_column):
    """
    Scatter plot with regression line.

    Args:
        df: DataFrame
        x_column: X-axis column
        y_column: Y-axis column
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(df[x_column], df[y_column], alpha=0.5)

    # Regression line
    from scipy.stats import linregress
    mask = ~np.isnan(df[x_column]) & ~np.isnan(df[y_column])
    x_clean = df[x_column][mask]
    y_clean = df[y_column][mask]

    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    line = slope * x_clean + intercept
    ax.plot(x_clean, line, 'r', label=f'R² = {r_value**2:.3f}')

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'{y_column} vs {x_column}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_pairplot(df, columns=None, hue=None):
    """
    Create pairplot for multiple columns.

    Args:
        df: DataFrame
        columns: List of columns to include (if None, use all numerical)
        hue: Column for color coding
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    fig = sns.pairplot(df[columns + ([hue] if hue else [])], hue=hue, diag_kind='kde')
    return fig


def plot_missing_values(df):
    """
    Visualize missing values in DataFrame.

    Args:
        df: DataFrame
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("No missing values found!")
        return None

    missing_pct = (missing / len(df)) * 100

    fig, ax = plt.subplots(figsize=(10, max(6, len(missing) * 0.3)))
    missing_pct.plot(kind='barh', ax=ax, color='salmon', edgecolor='black')
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Missing Values by Column')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    return fig


def plot_feature_importance(importances, feature_names, top_n=20):
    """
    Plot feature importance.

    Args:
        importances: Array of importance values
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    return fig


def plot_learning_curve(train_scores, val_scores, train_sizes=None):
    """
    Plot learning curve.

    Args:
        train_scores: Training scores over time/iterations
        val_scores: Validation scores over time/iterations
        train_sizes: Training set sizes (if applicable)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_axis = train_sizes if train_sizes is not None else range(len(train_scores))

    ax.plot(x_axis, train_scores, 'o-', label='Training Score', linewidth=2)
    ax.plot(x_axis, val_scores, 's-', label='Validation Score', linewidth=2)
    ax.set_xlabel('Training Size' if train_sizes is not None else 'Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_target_distribution(y, title='Target Distribution'):
    """
    Plot distribution of target variable.

    Args:
        y: Target variable (Series or array)
        title: Plot title
    """
    if isinstance(y, pd.Series):
        value_counts = y.value_counts()
    else:
        unique, counts = np.unique(y, return_counts=True)
        value_counts = pd.Series(counts, index=unique)

    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Add percentages on bars
    total = value_counts.sum()
    for i, v in enumerate(value_counts):
        ax.text(i, v + total*0.01, f'{v}\n({v/total*100:.1f}%)',
                ha='center', va='bottom')

    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
