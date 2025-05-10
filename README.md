Insurance Fraud Detection Using Decision Trees and Random Forests
Project Overview
This project focuses on detecting insurance fraud using machine learning techniques, specifically Decision Tree and Random Forest classifiers. The goal is to build and optimize predictive models to identify fraudulent insurance claims accurately. The project implements advanced hyperparameter tuning techniques, including Grid Search, Random Search, and Bayesian Optimization, to enhance model performance.
The dataset used consists of training and testing sets (Insurance Fraud - TRAIN-3000.csv and Insurance Fraud - TEST-12900.csv), with features processed to handle numerical and categorical data. Models are evaluated using four key metrics: Accuracy, Precision, Recall, and F1-Score, allowing for a comprehensive comparison of performance.
Key Features

Data Preprocessing:
Numerical features: Mean imputation and standard scaling.
Categorical features: Mode imputation and one-hot encoding.


Models:
Decision Tree Classifier
Random Forest Classifier


Hyperparameter Tuning:
Grid Search: Exhaustive search over specified parameter grids.
Random Search: Randomized sampling of hyperparameters.
Bayesian Optimization: Probabilistic model-based optimization.


Hyperparameters Tuned:
Decision Tree: max_depth, min_samples_split, min_samples_leaf.
Random Forest: n_estimators, max_depth, min_samples_split.


Evaluation Metrics:
Accuracy
Precision
Recall
F1-Score (weighted)


Cross-Validation: 5-fold cross-validation to prevent overfitting.

Results

Best Decision Tree Model:
Accuracy: 85.11%
Precision: 93.92%
Recall: 85.11%
F1-Score: 88.93%


Best Random Forest Model:
Accuracy: 95.78%
Precision: 96.83%
Recall: 95.78%
F1-Score: 96.20%



The Random Forest models consistently outperformed Decision Trees across all metrics, with the Grid Search and Bayesian Optimization methods yielding the best results.
Project Structure

Data:
Insurance Fraud - TRAIN-3000.csv: Training dataset with 3,000 records.
Insurance Fraud - TEST-12900.csv: Test dataset with 12,918 records.


Notebook:
Kalpesh_Patil_CIS508_Assigment2.ipynb: Jupyter Notebook containing the full code for data preprocessing, model training, hyperparameter tuning, and evaluation.


Presentation:
machine_learning_A2.pptx: PowerPoint slides summarizing the project methodology, results, and conclusions.


Scripts:
The notebook includes Python code using libraries like pandas, scikit-learn, and skopt for model development and tuning.



Installation and Setup

Clone the repository:git clone <repository-url>


Install required dependencies:pip install pandas scikit-learn skopt


Ensure the datasets (Insurance Fraud - TRAIN-3000.csv and Insurance Fraud - TEST-12900.csv) are placed in the project directory.
Run the Jupyter Notebook:jupyter notebook Kalpesh_Patil_CIS508_Assigment2.ipynb



Future Improvements

Data Quality: Address potential biases in the dataset.
Scalability: Optimize for larger datasets.
Real-time Processing: Adapt models for real-time fraud detection.
Interpretability: Enhance model explainability for stakeholder trust.
Cybersecurity: Strengthen data privacy measures.
Model Monitoring: Implement continuous monitoring to adapt to new fraud patterns.

Challenges

Balancing model complexity and interpretability.
Integrating models with existing insurance systems.
Ensuring robust performance against evolving fraud tactics.

Learning Outcomes
This project builds on a previous assignment by emphasizing advanced hyperparameter tuning and cross-validation. Key takeaways include:

Proficiency in Grid, Random, and Bayesian search for hyperparameter optimization.
Deeper understanding of ensemble methods like Random Forest.
Improved model evaluation through multiple metrics and cross-validation.

Conclusion
This project demonstrates a robust approach to insurance fraud detection using Decision Trees and Random Forests, with a focus on hyperparameter tuning to maximize performance. The Random Forest model, particularly when tuned with Grid Search or Bayesian Optimization, offers superior accuracy and reliability for fraud detection tasks.
Acknowledgments

Kalpesh Patil, for developing the project as part of CIS508 Assignment 2.
Libraries: scikit-learn, pandas, skopt for enabling efficient machine learning workflows.

