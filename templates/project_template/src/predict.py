"""
Prediction script template for ML projects.

This script loads a trained model and makes predictions on new data.

Usage:
    python predict.py --model models/model.pkl --input data/new_data.csv
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np


def load_model(model_path):
    """Load trained model from disk."""
    print(f"Loading model from {model_path}...")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("Model loaded successfully!")
    return model


def load_input_data(data_path):
    """Load input data for prediction."""
    print(f"Loading input data from {data_path}...")

    df = pd.read_csv(data_path)
    print(f"Loaded {df.shape[0]} samples for prediction")

    return df


def preprocess_input(df):
    """
    Preprocess input data.
    Should match preprocessing done during training.
    """
    print("Preprocessing input data...")

    # TODO: Apply same preprocessing as in training
    # Make sure to use the same transformations!

    return df


def make_predictions(model, X):
    """Make predictions using trained model."""
    print("Making predictions...")

    predictions = model.predict(X)

    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        return predictions, probabilities

    return predictions, None


def save_predictions(predictions, probabilities, output_path):
    """Save predictions to CSV file."""
    print(f"Saving predictions to {output_path}...")

    # Create output dataframe
    results = pd.DataFrame({
        'prediction': predictions
    })

    # Add probabilities if available
    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            results[f'probability_class_{i}'] = probabilities[:, i]

    # Save to CSV
    results.to_csv(output_path, index=False)
    print(f"Predictions saved! Total: {len(predictions)}")


def predict_single(model, features):
    """
    Make prediction for a single sample.

    Args:
        model: Trained model
        features: Dictionary or list of feature values

    Returns:
        prediction: Single prediction
    """
    # Convert to DataFrame if dict
    if isinstance(features, dict):
        X = pd.DataFrame([features])
    else:
        X = pd.DataFrame([features])

    # Preprocess
    X = preprocess_input(X)

    # Predict
    prediction = model.predict(X)[0]

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(X)[0]
        return prediction, probability

    return prediction, None


def main():
    """Main prediction pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input data CSV')
    parser.add_argument('--output', type=str, default='results/predictions.csv',
                        help='Path to save predictions')

    args = parser.parse_args()

    print("=" * 70)
    print("  PREDICTION PIPELINE")
    print("=" * 70)

    # Load model
    model = load_model(args.model)

    # Load input data
    df = load_input_data(args.input)

    # Preprocess input
    X = preprocess_input(df)

    # Make predictions
    predictions, probabilities = make_predictions(model, X)

    # Save predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, probabilities, output_path)

    # Display sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"  Sample {i+1}: {predictions[i]}", end="")
        if probabilities is not None:
            print(f" (confidence: {probabilities[i].max():.2%})")
        else:
            print()

    print("\n" + "=" * 70)
    print("  PREDICTION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
