# --- Import Libraries ---
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from xgboost import plot_importance

# --- Load Data ---
folder = r'C:\Users\User\Desktop\6th semester (spring 2025)\FDA\Kagel'
train_data = pd.read_csv(os.path.join(folder, 'fda_trainingset.csv'))
test_data = pd.read_csv(os.path.join(folder, 'fda_testset.csv'))
sample_submission_path = os.path.join(folder, 'sample_submission_FDA_file.csv')

try:
    submission_template = pd.read_csv(sample_submission_path)
    id_col = submission_template.columns[0]
    target_col = submission_template.columns[1]
except FileNotFoundError:
    print("Sample submission file not found. Defaulting ID and target columns.")
    id_col = 'ID'
    target_col = 'target'

# --- Prepare Features and Labels ---
features = train_data.drop(columns=['target'])
labels = train_data['target']

if id_col in test_data.columns:
    test_identifiers = test_data[id_col]
    test_features = test_data.drop(columns=[id_col])
else:
    test_identifiers = test_data.index
    test_features = test_data

# --- Preprocessing Pipeline ---
# 1. Merge training and test data
full_data = pd.concat([features, test_features], axis=0)

# 2. Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
full_imputed = pd.DataFrame(imputer.fit_transform(full_data), columns=full_data.columns)

# 3. Feature Scaling
scaler = StandardScaler()
full_scaled = pd.DataFrame(scaler.fit_transform(full_imputed), columns=full_imputed.columns)

# 4. Separate processed datasets
X_train_final = full_scaled.iloc[:len(features)]
X_test_final = full_scaled.iloc[len(features):]

# --- Train-Validation Split ---
X_train, X_valid, y_train, y_valid = train_test_split(X_train_final, labels, test_size=0.2, random_state=42)

# --- Build XGBoost Classifier ---
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.75,
    gamma=0.1,
    reg_alpha=0.3,
    reg_lambda=1,
    tree_method='hist',
    device='cuda',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)

# --- Model Evaluation ---
valid_preds = xgb_model.predict(X_valid)
valid_probs = xgb_model.predict_proba(X_valid)[:, 1]

print("\n=== Model Evaluation ===")
print(classification_report(y_valid, valid_preds))
print(f"AUC-ROC Score: {roc_auc_score(y_valid, valid_probs):.5f}")

# --- Feature Importance Plot ---
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, max_num_features=15, importance_type='gain')
plt.title("Feature Importance (Top 15)")
plt.tight_layout()
plt.show()

# --- Predictions on Test Set ---
test_predictions = xgb_model.predict_proba(X_test_final)[:, 1]

# --- Create Submission File ---
submission = pd.DataFrame({
    id_col: test_identifiers,
    target_col: test_predictions
})

submission_dir = os.path.join(folder, 'submissions')
os.makedirs(submission_dir, exist_ok=True)
submission_file = os.path.join(submission_dir, '21_submission.csv')
submission.to_csv(submission_file, index=False)

print(f"✅ Submission saved successfully: {submission_file}")
