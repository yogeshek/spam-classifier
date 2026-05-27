"""
Setup script to create the ML learning journey project structure.

Run this script to automatically create all necessary directories and template files.

Usage:
    python setup_structure.py
"""

import os
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def create_directory_structure():
    """Create the complete directory structure for ML learning journey."""

    base_dir = Path(__file__).parent

    # Define directory structure
    directories = [
        # Phase 1: Fundamentals
        "phase1_fundamentals/regression/house_price_predictor",
        "phase1_fundamentals/regression/linear_regression_basics",
        "phase1_fundamentals/regression/polynomial_regression",
        "phase1_fundamentals/classification/churn_prediction",
        "phase1_fundamentals/classification/loan_default",
        "phase1_fundamentals/evaluation/metrics_deep_dive",

        # Phase 2: Intermediate
        "phase2_intermediate/ensemble/xgboost_tutorial",
        "phase2_intermediate/ensemble/kaggle_competition",
        "phase2_intermediate/unsupervised/customer_segmentation",
        "phase2_intermediate/unsupervised/pca_dimensionality",
        "phase2_intermediate/advanced_nlp/word_embeddings",
        "phase2_intermediate/advanced_nlp/topic_modeling",

        # Phase 3: Deep Learning
        "phase3_deep_learning/neural_networks/nn_from_scratch",
        "phase3_deep_learning/computer_vision/image_classifier",
        "phase3_deep_learning/computer_vision/transfer_learning",
        "phase3_deep_learning/deep_nlp/text_generation",
        "phase3_deep_learning/deep_nlp/chatbot",

        # Phase 4: Advanced
        "phase4_advanced/recommender_systems",
        "phase4_advanced/time_series",
        "phase4_advanced/gans_autoencoders",
        "phase4_advanced/mlops",

        # Completed projects
        "completed_projects/spam_classifier",
        "completed_projects/sentiment_analysis",

        # Datasets
        "datasets/raw",
        "datasets/processed",
        "datasets/external",

        # Notebooks
        "notebooks/exploration",
        "notebooks/experiments",
        "notebooks/tutorials",

        # Utils
        "utils",

        # Templates
        "templates/project_template/data",
        "templates/project_template/notebooks",
        "templates/project_template/src",
        "templates/project_template/models",
        "templates/project_template/results",

        # Docs
        "docs/learning_notes",
        "docs/paper_summaries",

        # Tests
        "tests",
    ]

    print("Creating directory structure...")
    created_count = 0

    for directory in directories:
        dir_path = base_dir / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[+] Created: {directory}")
            created_count += 1
        else:
            print(f"[-] Already exists: {directory}")

    print(f"\n[SUCCESS] Directory structure setup complete!")
    print(f"          Created {created_count} new directories")

    # Create .gitkeep files in empty directories
    create_gitkeep_files(base_dir, directories)

    return created_count


def create_gitkeep_files(base_dir, directories):
    """Create .gitkeep files to track empty directories in git."""
    print("\nCreating .gitkeep files for empty directories...")

    for directory in directories:
        dir_path = base_dir / directory
        gitkeep_path = dir_path / ".gitkeep"

        # Only create .gitkeep if directory is empty
        if dir_path.exists() and not any(dir_path.iterdir()):
            gitkeep_path.touch()
            print(f"[+] Created .gitkeep in {directory}")


def create_readme_files():
    """Create README files for each phase."""

    base_dir = Path(__file__).parent

    readmes = {
        "phase1_fundamentals/README.md": """# Phase 1: Strengthen Fundamentals

**Duration**: 2-3 weeks
**Focus**: Core supervised learning algorithms

## Goals
- Master regression and classification algorithms
- Learn proper model evaluation techniques
- Work with structured data

## Projects
1. House Price Predictor (Regression)
2. Customer Churn Predictor (Classification)

## Key Concepts
- Linear/Polynomial Regression
- Logistic Regression
- Decision Trees & Random Forest
- Cross-validation
- Confusion Matrix, ROC-AUC

## Progress Tracker
- [ ] Linear Regression Basics
- [ ] House Price Predictor
- [ ] Churn Prediction
- [ ] Model Evaluation Deep Dive
""",

        "phase2_intermediate/README.md": """# Phase 2: Intermediate ML

**Duration**: 3-4 weeks
**Focus**: Advanced algorithms & feature engineering

## Goals
- Master ensemble methods
- Learn unsupervised learning
- Advanced text processing

## Projects
1. Kaggle Competition Entry (Ensemble)
2. Customer Segmentation (Clustering)
3. Document Clustering (Advanced NLP)

## Key Concepts
- XGBoost, LightGBM
- K-Means, PCA
- Word2Vec, GloVe
- Topic Modeling

## Progress Tracker
- [ ] XGBoost Tutorial
- [ ] Kaggle Competition
- [ ] Customer Segmentation
- [ ] Word Embeddings
""",

        "phase3_deep_learning/README.md": """# Phase 3: Deep Learning Basics

**Duration**: 4-6 weeks
**Focus**: Neural networks & modern architectures

## Goals
- Understand neural networks deeply
- Master CNNs for computer vision
- Learn RNNs/Transformers for NLP

## Projects
1. Neural Network from Scratch
2. Image Classifier (CNNs)
3. Text Generator/Chatbot (RNN/Transformer)

## Key Concepts
- Neural Networks, Backpropagation
- CNNs, Transfer Learning
- RNN, LSTM, Transformers
- PyTorch/TensorFlow

## Progress Tracker
- [ ] Build NN from Scratch
- [ ] Image Classifier
- [ ] Text Generation
- [ ] Chatbot
""",

        "phase4_advanced/README.md": """# Phase 4: Advanced & Specialization

**Duration**: Ongoing
**Focus**: Specialized topics & production ML

## Goals
- Master specialized ML domains
- Learn to deploy models
- MLOps practices

## Projects
1. Movie Recommender System
2. Stock Price Forecasting
3. GAN Image Generation
4. Deploy a Model to Production

## Key Concepts
- Collaborative Filtering
- Time Series (ARIMA, Prophet)
- GANs, Autoencoders
- Docker, CI/CD, Monitoring

## Progress Tracker
- [ ] Recommender System
- [ ] Time Series Forecasting
- [ ] GAN Project
- [ ] MLOps & Deployment
""",
    }

    print("\nCreating README files for each phase...")

    for file_path, content in readmes.items():
        full_path = base_dir / file_path
        if not full_path.exists():
            full_path.write_text(content)
            print(f"[+] Created: {file_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("  ML LEARNING JOURNEY - PROJECT STRUCTURE SETUP")
    print("=" * 70)
    print()

    # Create directory structure
    create_directory_structure()

    # Create README files
    create_readme_files()

    print("\n" + "=" * 70)
    print("  [SUCCESS] SETUP COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the PROJECT_STRUCTURE.md file")
    print("2. Move existing projects to completed_projects/")
    print("3. Check out the templates/ directory")
    print("4. Start with Phase 1: House Price Predictor")
    print("\nHappy learning!")
