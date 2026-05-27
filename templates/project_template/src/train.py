"""
Training script template for ML projects.

This script handles:
- Loading and preprocessing data
- Training the model
- Evaluating performance
- Saving the trained model

Usage:
    python train.py --data path/to/data.csv --model models/model.pkl
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import your model (example)
# from sklearn.ensemble import RandomForestClassifier


def load_data(data_path):
    """Load dataset from CSV file."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """
    Preprocess the data.

    Steps:
    - Handle missing values
    - Encode categorical variables
    - Feature engineering
    - Scale features
    """
    print("Preprocessing data...")

    # TODO: Implement preprocessing steps
    # Example:
    # df = df.dropna()
    # df = pd.get_dummies(df, columns=['categorical_col'])

    return df


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    print(f"Splitting data (test_size={test_size})...")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the machine learning model."""
    print("Training model...")

    # TODO: Initialize and train your model
    # Example:
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

    model = None  # Replace with your model

    print("Model training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set."""
    print("\nEvaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


def save_model(model, output_path):
    """Save trained model to disk."""
    print(f"\nSaving model to {output_path}...")

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved successfully!")


def main():
    """Main training pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data', type=str, default='data/processed/data.csv',
                        help='Path to training data')
    parser.add_argument('--target', type=str, default='target',
                        help='Name of target column')
    parser.add_argument('--model', type=str, default='models/model.pkl',
                        help='Path to save trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')

    args = parser.parse_args()

    print("=" * 70)
    print("  MODEL TRAINING PIPELINE")
    print("=" * 70)

    # Load data
    df = load_data(args.data)

    # Preprocess data
    df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df, args.target, test_size=args.test_size
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, Path(args.model))

    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE! Accuracy: {accuracy:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
