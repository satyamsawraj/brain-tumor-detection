import numpy as np
import joblib
from sklearn.svm import SVC

# Load data
X_train = np.load("X_train_dtcwt.npy")
y_train = np.load("y_train.npy")

# Train best model (SVM)
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "brain_tumor_svm_model.pkl")

print("✅ SVM model saved as brain_tumor_svm_model.pkl")
