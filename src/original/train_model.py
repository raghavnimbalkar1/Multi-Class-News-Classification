import pickle
import sys
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_config_path

# Load preprocessed feature matrices and labels
with open(get_config_path('model_artifacts.original.X_train'), "rb") as f:
    X_train = pickle.load(f)

with open(get_config_path('model_artifacts.original.X_test'), "rb") as f:
    X_test = pickle.load(f)

with open(get_config_path('model_artifacts.original.y_train'), "rb") as f:
    y_train = pickle.load(f)

with open(get_config_path('model_artifacts.original.y_test'), "rb") as f:
    y_test = pickle.load(f)

# Initialize SVM classifier (Linear SVM for high-dimensional data)
svm_clf = LinearSVC(max_iter=5000, random_state=42)

# Train the model
print("Training SVM classifier...")
svm_clf.fit(X_train, y_train)
print("Training complete.")

# Save the trained model
with open(get_config_path('model_artifacts.original.svm_model'), "wb") as f:
    pickle.dump(svm_clf, f)
print(f"Trained SVM model saved to {get_config_path('model_artifacts.original.svm_model')}")

# Predict on test set
y_pred = svm_clf.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
