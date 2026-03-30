import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

X = np.load("X_train_dtcwt.npy")
y = np.load("y_train.npy")

model = SVC(kernel="rbf")

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold)

print("✅ 10-Fold Cross Validation Accuracy")
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())
