# PCA-ML-Visualization

This project performs diabetes prediction using machine learning with PCA for data visualization.

## Dataset

The dataset is from Kaggle Playground Series S5E12: Binary Classification with a Tabular Insurance Dataset.

## Requirements

- Python 3.8-3.12 (avoid 3.14 alpha due to compatibility issues)
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn

Install with:
```
pip install -r requirements.txt
```

## Setup

Due to Python 3.14 alpha compatibility issues with numpy, use Python 3.11 or 3.12:

1. Install Python 3.11 from https://www.python.org/downloads/
2. Create venv: `python -m venv venv`
3. Activate: `.\venv\Scripts\activate`
4. Install packages: `pip install -r requirements.txt`

## Usage

1. Ensure data files are in `data/` directory: `train.csv`, `test.csv`, `sample_submission.csv`

2. Run the training script:
```
python scripts/train_model.py
```

This will:
- Load and preprocess the data
- Perform PCA visualization
- Train a Random Forest model
- Evaluate on validation set
- Generate predictions on test set
- Save submission to `data/submission.csv`

## Notebook

Alternatively, use the Jupyter notebook `notebooks/diabetes_prediction.ipynb` for interactive exploration.

## Results

- PCA plot saved as `pca_visualization.png`
- Model accuracy and metrics printed
- Submission file for Kaggle