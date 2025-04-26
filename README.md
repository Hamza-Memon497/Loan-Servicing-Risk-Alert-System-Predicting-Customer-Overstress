# Loan-Servicing-Risk-Alert-System-Predicting-Customer-Overstress
Project Overview
This project tackles the challenge of identifying overstressed microfinance clients using transactional, profile, and financial behavior data. Early detection enables financial institutions to offer counseling or restructuring support, preventing defaults and supporting financial inclusion.
The task involved building a binary classification model to predict customer overstress (1 = overstressed, 0 = not overstressed).

Key Evaluation Metrics:

Precision

Recall (special focus)

AUC-ROC

My Approach
Data Preprocessing:

Missing values handled using median imputation via SimpleImputer.

Feature scaling applied using StandardScaler.

Combined training and testing sets during preprocessing for consistent transformations.

Model Building:

Used XGBoost Classifier with customized hyperparameters:

200 trees (n_estimators)

Learning rate of 0.03

Max depth of 7

Subsampling and column sampling to prevent overfitting

Regularization (alpha = 0.3, lambda = 1)

GPU acceleration (device='cuda') for faster training

Evaluation performed on an 80/20 Train-Validation split.

Model Evaluation:

Achieved 92.8% accuracy on validation data.

High recall and AUC-ROC scores, aligning with project objectives.

Plotted top 15 feature importances using plot_importance.

Submission:

Predictions were made on the test set using the trained model.

Submission file generated in the required format.

Results
Validation Accuracy: 92.8%

Strong Model Performance: Balanced sensitivity and specificity; prioritized recall as per project guidelines.

Feature Importance: Provided insights into the key factors affecting customer overstress.

Repository Structure
bash
Copy
Edit
├── fda_trainingset.csv               # Training data with features and labels
├── fda_testset.csv                   # Test data (features only)
├── sample_submission_FDA_file.csv    # Sample format for submissions
├── 21_submission.csv                 # Final submission file
├── main_notebook.ipynb               # Well-commented notebook containing full code
└── README.md                         # This document
Tools and Libraries
Python 3

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib

