import numpy as np
import joblib
from sklearn.svm import SVC

# Load multi-class data
X_train = np.load("X_train_mc_dtcwt.npy")
y_train = np.load("y_train_mc.npy")

# Train SVM (multi-class)
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "brain_tumor_multiclass_model.pkl")

print("✅ Multi-class model saved successfully")
