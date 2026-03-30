import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Load
X_test = np.load("X_test_all.npy")
y_test = np.load("y_test_mc.npy")
model = joblib.load("brain_tumor_final_model.pkl")

classes = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
y_test_bin = label_binarize(y_test, classes=[0,1,2,3])
y_score = model.predict_proba(X_test)

plt.figure(figsize=(7,6))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend()
plt.show()
