"""
Shared data preprocessing utilities for ML projects.

Common preprocessing functions that can be reused across different projects.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def handle_missing_values(df, strategy='mean', categorical_strategy='most_frequent'):
    """
    Handle missing values in numerical and categorical columns.

    Args:
        df: Input DataFrame
        strategy: Strategy for numerical columns ('mean', 'median', 'constant')
        categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')

    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()

    # Numerical columns
    numerical_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_cols) > 0:
        imputer = SimpleImputer(strategy=strategy)
        df_copy[numerical_cols] = imputer.fit_transform(df_copy[numerical_cols])

    # Categorical columns
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        imputer = SimpleImputer(strategy=categorical_strategy)
        df_copy[categorical_cols] = imputer.fit_transform(df_copy[categorical_cols])

    return df_copy


def encode_categorical_features(df, columns=None, method='onehot'):
    """
    Encode categorical features.

    Args:
        df: Input DataFrame
        columns: List of columns to encode (if None, encode all object columns)
        method: 'onehot' or 'label'

    Returns:
        DataFrame with encoded features
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['object']).columns.tolist()

    if method == 'onehot':
        df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
    elif method == 'label':
        for col in columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))

    return df_copy


def scale_features(df, columns=None, method='standard'):
    """
    Scale numerical features.

    Args:
        df: Input DataFrame
        columns: List of columns to scale (if None, scale all numerical columns)
        method: 'standard' or 'minmax'

    Returns:
        DataFrame with scaled features, scaler object
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")

    df_copy[columns] = scaler.fit_transform(df_copy[columns])

    return df_copy, scaler


def remove_outliers(df, columns=None, method='iqr', threshold=1.5):
    """
    Remove outliers from numerical columns.

    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with outliers removed
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if method == 'iqr':
        for col in columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]

    elif method == 'zscore':
        from scipy import stats
        for col in columns:
            z_scores = np.abs(stats.zscore(df_copy[col]))
            df_copy = df_copy[z_scores < threshold]

    return df_copy


def create_train_test_split(df, target_column, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def feature_engineering_datetime(df, datetime_column):
    """
    Extract features from datetime column.

    Args:
        df: Input DataFrame
        datetime_column: Name of datetime column

    Returns:
        DataFrame with additional datetime features
    """
    df_copy = df.copy()

    # Convert to datetime if needed
    if df_copy[datetime_column].dtype != 'datetime64[ns]':
        df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])

    # Extract features
    df_copy[f'{datetime_column}_year'] = df_copy[datetime_column].dt.year
    df_copy[f'{datetime_column}_month'] = df_copy[datetime_column].dt.month
    df_copy[f'{datetime_column}_day'] = df_copy[datetime_column].dt.day
    df_copy[f'{datetime_column}_dayofweek'] = df_copy[datetime_column].dt.dayofweek
    df_copy[f'{datetime_column}_hour'] = df_copy[datetime_column].dt.hour
    df_copy[f'{datetime_column}_is_weekend'] = df_copy[datetime_column].dt.dayofweek.isin([5, 6]).astype(int)

    return df_copy


def get_data_summary(df):
    """
    Get comprehensive summary of DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numerical_summary': df.describe().to_dict(),
        'categorical_summary': {
            col: df[col].value_counts().to_dict()
            for col in df.select_dtypes(include=['object']).columns
        }
    }

    return summary


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing utilities loaded successfully!")
