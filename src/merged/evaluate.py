import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. Load the New Model and Test Data (Updated Paths) ---
print("Step 1: Loading model and test data from '/Models/Merged/'...")
with open("../../models/merged/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("../../models/merged/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

with open("../../models/merged/svm_model.pkl", "rb") as f:
    svm_clf = pickle.load(f)
print("Loading complete.")


#  2. Make Predictions (Unchanged Logic) 
print("Step 2: Making predictions on the test set...")
y_pred = svm_clf.predict(X_test)


# 3. Generate and Save Reports (Updated Paths)
print("Step 3: Generating and saving classification reports...")
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy: {accuracy}")

# Detailed classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Save classification report as CSV
report_path = "../../models/merged/classification_report.csv"
df_report = pd.DataFrame(report).transpose()
df_report.to_csv(report_path, index=True)
print(f"Classification report saved to {report_path}")

# 4. Generate and Save Confusion Matrices (Updated Paths)
print("Step 4: Generating and saving confusion matrices...")
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y_test))) 

# Save visual confusion matrix
plt.figure(figsize=(10, 8)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (Improved Model - 13 Classes)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path_png = "../../models/merged/confusion_matrix_evaluate.png"
plt.savefig(cm_path_png)
print(f"Visual confusion matrix saved to {cm_path_png}")

# Save numeric confusion matrix as CSV
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_path_csv = "../../models/merged/confusion_matrix_numeric.csv"
cm_df.to_csv(cm_path_csv)
print(f"Numeric confusion matrix saved to {cm_path_csv}")

# 5. Find Top Misclassifications (Unchanged Logic) 
print("\n--- Top 10 Misclassifications for the Improved Model ---")
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)
misclassified_pairs = np.unravel_index(np.argsort(cm_off_diag.ravel())[::-1], cm_off_diag.shape)
for i, j in zip(*misclassified_pairs[:10]):
    if i < len(labels) and j < len(labels):
        print(f"Actual: {labels[i]}, Predicted: {labels[j]}, Count: {cm[i, j]}")

print("\nEvaluation complete. All reports are saved in '/Models/Merged/'.")