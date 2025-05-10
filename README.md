# Insurance Fraud Detection Using Decision Trees and Random Forests

## Project Overview
This project focuses on detecting insurance fraud using machine learning techniques, specifically Decision Tree and Random Forest classifiers. The goal is to build and optimize predictive models to identify fraudulent insurance claims accurately. The project implements advanced hyperparameter tuning techniques, including Grid Search, Random Search, and Bayesian Optimization, to enhance model performance.

The dataset used consists of training and testing sets (`Insurance Fraud - TRAIN-3000.csv` and `Insurance Fraud - TEST-12900.csv`), with features processed to handle numerical and categorical data. Models are evaluated using four key metrics: Accuracy, Precision, Recall, and F1-Score, allowing for a comprehensive comparison of performance.

## Key Features
- **Data Preprocessing**:
  - Numerical features: Mean imputation and standard scaling.
  - Categorical features: Mode imputation and one-hot encoding.
- **Models**:
  - Decision Tree Classifier
  - Random Forest Classifier
- **Hyperparameter Tuning**:
  - Grid Search: Exhaustive search over specified parameter grids.
  - Random Search: Randomized sampling of hyperparameters.
  - Bayesian Optimization: Probabilistic model-based optimization.
- **Hyperparameters Tuned**:
  - Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leaf`.
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score (weighted)
- **Cross-Validation**: 5-fold cross-validation to prevent overfitting.

## Results
- **Best Decision Tree Model**:
  - Accuracy: 85.11%
  - Precision: 93.92%
  - Recall: 85.11%
  - F1-Score: 88.93%
- **Best Random Forest Model**:
  - Accuracy: 95.78%
  - Precision: 96.83%
  - Recall: 95.78%
  - F1-Score: 96.20%

The Random Forest models consistently outperformed Decision Trees across all metrics, with the Grid Search and Bayesian Optimization methods yielding the best results.

## Project Structure
- **Data**:
  - `Insurance Fraud - TRAIN-3000.csv`: Training dataset with 3,000 records.
  - `Insurance Fraud - TEST-12900.csv`: Test dataset with 12,918 records.
- **Scripts**:
  - The notebook includes Python code using libraries like `pandas`, `scikit-learn`, and `skopt` for model development and tuning.

**Thanks and feel free reach out for any details**
