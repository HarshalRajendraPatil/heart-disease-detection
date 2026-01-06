# Heart Disease Prediction using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

## Overview
This project is the first capstone from the ["Complete Machine Learning & Data Science Zero to Mastery" Udemy course](https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/). It builds a supervised machine learning model to predict whether a patient has heart disease (binary classification: 0 = no, 1 = yes) based on 13 medical attributes like age, sex, cholesterol, and exercise-induced angina.

Using the classic [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (303 samples), the model achieves **~85% accuracy** with a tuned Random Forest Classifier—highlighting the power of ensemble methods for medical diagnostics.

This serves as a foundational step toward Generative AI expertise, reinforcing core ML concepts like feature engineering and evaluation that underpin advanced models (e.g., transformers in LLMs).

## Key Features
- **End-to-End Pipeline**: Data loading, EDA, model training, hyperparameter tuning, evaluation, and export.
- **Models Compared**: Logistic Regression, K-Nearest Neighbors (KNN), Random Forest Classifier.
- **Best Model**: Random Forest (tuned via RandomizedSearchCV) with top features like chest pain type and ST depression.
- **Visualizations**: Bar plots for class balance/sex distribution, scatter for age vs. heart rate, correlation heatmap, model comparison bar, tuning curves, and feature importance.

## Tech Stack
- **Language**: Python 3.8+
- **Libraries**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - ML: Scikit-learn (models, splitting, tuning, metrics)
  - Export: Pickle
- **Environment**: Jupyter Notebook (tested on Python 3.13)

## Dataset
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Shape: 303 rows × 14 columns (13 features + 1 target)
- Features: age, sex, cp (chest pain), trestbps (resting blood pressure), chol (cholesterol), fbs (fasting blood sugar), restecg (ECG results), thalach (max heart rate), exang (exercise angina), oldpeak (ST depression), slope, ca (vessels), thal (thalassemia).
- Target: Binary (heart disease presence).

## Results
- Baseline Accuracies: Random Forest (85%), Logistic Regression (85%), KNN (83%).
- Tuned Best: Random Forest with 5-fold CV (~86% mean score).
- Evaluation: Accuracy, precision, recall, F1, ROC-AUC (via cross-validation).
- Feature Importance: Top predictors—chest pain (cp), ST depression (oldpeak), vessels (ca).

## How to Run
1. **Clone the Repo**: 
  git clone https://github.com/harshalrajendrapatil/heart-disease-detection.git
  cd heart-disease-prediction
2. **Install Dependencies** (via pip/conda):
  pip install numpy pandas seaborn scikit-learn
3. **Download Dataset**: 
  Place `heart-disease.csv` in the root (or update the notebook path).
4. **Run the Notebook**:
  jupyter notebook Heart-disease-detection.ipynb
