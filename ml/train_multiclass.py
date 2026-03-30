import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

X_train = np.load("X_train_mc_dtcwt.npy")
y_train = np.load("y_train_mc.npy")
X_test = np.load("X_test_mc_dtcwt.npy")
y_test = np.load("y_test_mc.npy")

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred,
    target_names=["No Tumor", "Glioma", "Meningioma", "Pituitary"]
))
