import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load test data
with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Load trained SVM model
with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/svm_model.pkl", "rb") as f:
    svm_clf = pickle.load(f)

# Predict on test set
y_pred = svm_clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save classification report as CSV
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("/home/skinny/Documents/Code/MultiClassNewsClassification/models/classification_report.csv", index=True)
print("Classification report saved as models/classification_report.csv")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14,10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save confusion matrix as PNG
plt.savefig("/home/skinny/Documents/Code/MultiClassNewsClassification/models/confusion_matrix_evaluate.png")
print("Confusion matrix saved as models/confusion_matrix_evaluate.png")


# y_true, y_pred are your test labels and predicted labels
cm = confusion_matrix(y_test, y_pred)
labels = sorted(list(set(y_test)))  # or your predefined label list

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv("/home/skinny/Documents/Code/MultiClassNewsClassification/models/confusion_matrix_numeric.csv")

# Find top misclassifications
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)
misclassified_pairs = np.unravel_index(np.argsort(cm_off_diag.ravel())[::-1], cm_off_diag.shape)
for i, j in zip(*misclassified_pairs[:10]):
    print(f"Actual: {labels[i]}, Predicted: {labels[j]}, Count: {cm[i, j]}")
