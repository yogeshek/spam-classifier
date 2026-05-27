# ML Learning Journey - Project Structure

## 📁 Directory Organization

```
spam-classifier/
│
├── phase1_fundamentals/          # Phase 1: Strengthen Fundamentals
│   ├── regression/
│   │   ├── house_price_predictor/
│   │   ├── linear_regression_basics/
│   │   └── polynomial_regression/
│   ├── classification/
│   │   ├── churn_prediction/
│   │   └── loan_default/
│   └── evaluation/
│       └── metrics_deep_dive/
│
├── phase2_intermediate/          # Phase 2: Intermediate ML
│   ├── ensemble/
│   │   ├── xgboost_tutorial/
│   │   └── kaggle_competition/
│   ├── unsupervised/
│   │   ├── customer_segmentation/
│   │   └── pca_dimensionality/
│   └── advanced_nlp/
│       ├── word_embeddings/
│       └── topic_modeling/
│
├── phase3_deep_learning/         # Phase 3: Deep Learning
│   ├── neural_networks/
│   │   └── nn_from_scratch/
│   ├── computer_vision/
│   │   ├── image_classifier/
│   │   └── transfer_learning/
│   └── deep_nlp/
│       ├── text_generation/
│       └── chatbot/
│
├── phase4_advanced/              # Phase 4: Advanced Topics
│   ├── recommender_systems/
│   ├── time_series/
│   ├── gans_autoencoders/
│   └── mlops/
│
├── completed_projects/           # Your completed work
│   ├── spam_classifier/          # Move existing spam work here
│   └── sentiment_analysis/       # Move existing sentiment work here
│
├── datasets/                     # All datasets
│   ├── raw/                      # Original unprocessed data
│   ├── processed/                # Cleaned and preprocessed data
│   └── external/                 # Downloaded from Kaggle, UCI, etc.
│
├── models/                       # Saved models (existing)
│   ├── spam_classifier/
│   └── sentiment_analysis/
│
├── notebooks/                    # Jupyter notebooks (existing)
│   ├── exploration/              # EDA notebooks
│   ├── experiments/              # Model experiments
│   └── tutorials/                # Learning notebooks
│
├── utils/                        # Shared utility functions
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── evaluation_metrics.py
│   └── model_utils.py
│
├── templates/                    # Project templates
│   ├── project_template/
│   ├── notebook_template.ipynb
│   └── README_template.md
│
├── docs/                         # Documentation
│   ├── learning_notes/
│   └── paper_summaries/
│
├── tests/                        # Unit tests
│
├── .gitignore
├── requirements.txt
├── ML_JOURNEY_ROADMAP.md         # Your roadmap (existing)
└── README.md
```

## 🎯 Each Project Should Have

Every project folder should contain:
```
project_name/
├── data/                         # Project-specific data
├── notebooks/                    # Jupyter notebooks for this project
├── src/                          # Source code
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/                       # Saved models
├── results/                      # Outputs, plots, metrics
├── README.md                     # Project documentation
└── requirements.txt              # Project-specific dependencies
```

## 📝 Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`
- **Notebooks**: `01_descriptive_name.ipynb` (numbered for order)

## 🔧 Setup Instructions

1. Create the directory structure (automated script provided)
2. Move existing projects to `completed_projects/`
3. Set up virtual environment for each phase
4. Use templates for new projects
5. Track progress in ML_JOURNEY_ROADMAP.md
