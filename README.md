# Advanced Feature Selection and Feature Engineering for Wine Classification

A comprehensive feature selection and engineering pipeline applied to the UCI Wine
Classification dataset. Multiple feature selection strategies are systematically
compared to identify the optimal feature subset for predicting wine classes using
a Gradient Boosting classifier.

## Problem Statement

Given 13 physicochemical measurements of wines from three Italian cultivars, predict
the wine class (0, 1, or 2) and determine which feature selection strategy produces
the most accurate and generalisable model.

## Dataset

- Source: sklearn.datasets.load_wine (UCI Wine Recognition Dataset)
- Shape: 178 samples x 13 features
- Target: 3 wine classes — Class 0 (59), Class 1 (71), Class 2 (48)
- Features: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols,
  flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue,
  od280/od315_of_diluted_wines, proline

## Pipeline Overview

1. Exploratory data analysis — class distribution and feature correlation heatmap
2. Stratified 60/20/20 train/validation/test split
3. Baseline model — GradientBoostingClassifier on all 13 features
4. Five feature selection methods evaluated and compared
5. Feature engineering — four new interaction and ratio features created
6. All strategies compared by weighted F1 on the validation set
7. Best strategy validated using 5-fold stratified cross-validation
8. Final model trained on train + validation and evaluated on held-out test set

## Feature Selection Methods

| Method | Description |
|---|---|
| Variance Threshold | Removes features with variance below 0.01 after MinMax scaling |
| SelectKBest (Mutual Information) | Selects top k features by mutual information score — k tuned from 1 to 13 |
| RFE (Recursive Feature Elimination) | Eliminates features step-by-step using RandomForest importance — k tuned from 1 to 13 |
| Correlation Filtering | Removes one feature from each pair with correlation above 0.85 |
| Random Forest Importance | Selects top 5 features by Gini importance |
| Permutation Importance | Selects top 5 features by permutation importance on validation set |

## Feature Engineering

Four new features were created and tested alongside all original features:

| Feature | Formula |
|---|---|
| phenols_flavanoids_ratio | total_phenols / (flavanoids + 1e-6) |
| alcohol_proline_interaction | alcohol x proline |
| color_hue_interaction | color_intensity x hue |
| malic_magnesium_ratio | malic_acid / (magnesium + 1e-6) |

## Tech Stack

| Library | Purpose |
|---|---|
| scikit-learn | Feature selection, classifiers, cross-validation |
| pandas / numpy | Data manipulation |
| matplotlib | Visualisation — bar charts, correlation heatmap, confusion matrices |
