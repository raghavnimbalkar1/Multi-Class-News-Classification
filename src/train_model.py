import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load preprocessed feature matrices and labels
with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Initialize SVM classifier (Linear SVM for high-dimensional data)
svm_clf = LinearSVC(max_iter=5000, random_state=42)

# Train the model
print("Training SVM classifier...")
svm_clf.fit(X_train, y_train)
print("Training complete.")

# Save the trained model
with open("/home/skinny/Documents/Code/MultiClassNewsClassification/models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_clf, f)
print("Trained SVM model saved as models/svm_model.pkl")

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
