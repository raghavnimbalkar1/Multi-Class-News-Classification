
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1. Load Model, Test Data, and Encoder 
print("Step 1: Loading XGBoost model, test data, and encoder...")
with open("../../models/merged/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("../../models/merged/y_test.pkl", "rb") as f:
    y_test = pickle.load(f) # The TRUE labels (strings)

with open("../../models/merged/xgboost_model.pkl", "rb") as f:
    xgb_clf = pickle.load(f)

with open("../../models/merged/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
print("Loading complete.")


# 2. Make Predictions 
print("Step 2: Making predictions on the test set...")
# Model predicts encoded labels (0, 1, 2...)
y_pred_encoded = xgb_clf.predict(X_test)

y_pred = encoder.inverse_transform(y_pred_encoded)
print("Predictions complete and decoded.")


# 3. Generate and Save Reports 
print("Step 3: Generating and saving classification reports for XGBoost...")
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy}")

# classification report (comparing text labels to text labels)
report = classification_report(y_test, y_pred, output_dict=True)

# Save classification report as CSV
report_path = "../../models/merged/xgboost_classification_report.csv"
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(report_path, index=True)
print(f"XGBoost classification report saved to {report_path}")

#4. Generate and Save Confusion Matrices 
print("Step 4: Generating and saving XGBoost confusion matrices...")
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y_test))) # Get the new 13 category names

# Save visual confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (XGBoost Model - 13 Classes)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path_png = "../../models/merged/xgboost_confusion_matrix_evaluate.png"
plt.savefig(cm_path_png)
print(f"Visual confusion matrix saved to {cm_path_png}")

# Save numeric confusion matrix as CSV
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_path_csv = "../../models/merged/xgboost_confusion_matrix_numeric.csv"
cm_df.to_csv(cm_path_csv)
print(f"Numeric confusion matrix saved to {cm_path_csv}")

print("\nXGBoost evaluation complete. All reports are saved in '/models/merged/'.")