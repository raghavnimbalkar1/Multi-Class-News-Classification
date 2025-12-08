import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load Merged Feature Matrices (Updated Paths) ---
print("Step 1: Loading data from '/Models/Merged/'...")
with open("../../models/merged/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)

with open("../../models/merged/y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("../../models/merged/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("../../models/merged/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)
print("Data loading complete.")


# --- 2. Initialize Improved SVM Classifier (The New Step) ---
print("Step 2: Initializing LinearSVC model with class_weight='balanced'...")
svm_clf = LinearSVC(max_iter=5000, random_state=42, class_weight='balanced')


# --- 3. Train the Model (Unchanged Logic) ---
print("Step 3: Training the improved SVM classifier...")
svm_clf.fit(X_train, y_train)
print("Training complete.")


# --- 4. Save the Trained Model (Updated Path) ---
print("Step 4: Saving the trained model...")
output_path = "../../models/merged/svm_model.pkl"
with open(output_path, "wb") as f:
    pickle.dump(svm_clf, f)
print(f"Trained SVM model saved to {output_path}")


# --- 5. Quick Evaluation Check (For Convenience) ---
print("\n--- Quick Performance Check ---")
y_pred = svm_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Improved SVM Confusion Matrix (Quick Check)")
plt.show()

print("\nTraining script finished.")