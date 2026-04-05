import pickle
import sys
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config_path

# Load test data
with open(get_config_path('model_artifacts.original.X_test'), "rb") as f:
    X_test = pickle.load(f)

with open(get_config_path('model_artifacts.original.y_test'), "rb") as f:
    y_test = pickle.load(f)

# Load trained SVM model
with open(get_config_path('model_artifacts.original.svm_model'), "rb") as f:
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
df_report.to_csv(get_config_path('model_artifacts.original.classification_report'), index=True)
print(f"Classification report saved to {get_config_path('model_artifacts.original.classification_report')}")

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
cm_df.to_csv(get_config_path('model_artifacts.original.confusion_matrix'), index=True)
print(f"Numeric confusion matrix saved to {get_config_path('model_artifacts.original.confusion_matrix')}")
