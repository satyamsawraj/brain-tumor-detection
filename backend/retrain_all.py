import os
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
import cv2
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

# ── PATH SETUP ───────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath("../ml"))
from advanced_features import extract_features

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_DIR = "../dataset/Training"
IMG_SIZE    = 224
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ── LOAD DATASET ─────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
X, y = [], []

for label, cls in enumerate(CLASSES):
    class_dir = os.path.join(DATASET_DIR, cls)
    files     = os.listdir(class_dir)
    print(f"   Loading {cls}: {len(files)} images...")
    for file in files:
        path = os.path.join(class_dir, file)
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        features = extract_features(img, use_ltp=True)
        if isinstance(features, tuple):
            features = features[0]
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f"\n✅ Dataset loaded: {X.shape}")

# ── SPLIT ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── DIMENSIONALITY REDUCTION ─────────────────────────────────────────────────
print("\n🔬 Applying PCA, LDA, NCA...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pca = PCA(n_components=20)
lda = LDA(n_components=3)
nca = NCA(max_iter=500)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda  = lda.transform(X_test_scaled)
X_train_nca = nca.fit_transform(X_train_scaled, y_train)
X_test_nca  = nca.transform(X_test_scaled)

X_train_final = np.hstack([X_train_pca, X_train_lda, X_train_nca])
X_test_final  = np.hstack([X_test_pca,  X_test_lda,  X_test_nca])
print(f"✅ Combined feature shape: {X_train_final.shape}")

# ── SAVE TRANSFORMERS ─────────────────────────────────────────────────────────
print("\n💾 Saving transformers...")
joblib.dump(scaler, "../ml/feature_scaler.pkl")
joblib.dump(pca,    "../ml/feature_pca.pkl")
joblib.dump(lda,    "../ml/feature_lda.pkl")
joblib.dump(nca,    "../ml/feature_nca.pkl")
print("✅ Transformers saved!")

# ── TRAIN MODELS ─────────────────────────────────────────────────────────────
models_to_train = {
    "SVM": SVC(probability=True, kernel='rbf', C=10),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RF":  RandomForestClassifier(n_estimators=150)
}

results     = {}
roc_data    = {}
cm_data     = {}
report_data = {}

for name, model in models_to_train.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train_final, y_train)
    preds = model.predict(X_test_final)
    acc   = accuracy_score(y_test, preds)
    print(f"✅ {name} Accuracy: {acc*100:.2f}%")

    report = classification_report(y_test, preds, target_names=CLASS_NAMES)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    cm_data[name] = cm
    print("Confusion Matrix:\n", cm)

    # ROC
    y_bin  = label_binarize(y_test, classes=[0, 1, 2, 3])
    proba  = model.predict_proba(X_test_final)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc[i]         = auc(fpr[i], tpr[i])
    roc_data[name]    = (fpr, tpr, roc_auc)
    results[name]     = acc
    report_data[name] = acc

    joblib.dump(model, f"../ml/{name.lower()}_model.pkl")
    print(f"💾 Saved: {name.lower()}_model.pkl")

# ── SAVE AS STREAMLIT MODEL NAMES ────────────────────────────────────────────
print("\n💾 Saving named models for Streamlit...")
joblib.dump(models_to_train["SVM"], "../ml/brain_tumor_advanced_model.pkl")
joblib.dump(models_to_train["KNN"], "../ml/svm_meta_model.pkl")
joblib.dump(models_to_train["RF"],  "../ml/svm_woa_model.pkl")

# ── SAVE ROC + CM DATA FOR STREAMLIT ─────────────────────────────────────────
print("\n💾 Saving ROC and CM data for Streamlit...")
eval_data = {
    "roc":      roc_data,
    "cm":       cm_data,
    "accuracy": results,
    "classes":  CLASS_NAMES
}
joblib.dump(eval_data, "../ml/eval_data.pkl")
print("✅ eval_data.pkl saved!")

# ── COMBINED ROC PLOT ─────────────────────────────────────────────────────────
os.makedirs("../static/outputs", exist_ok=True)
colors_roc = {"SVM": "#00d4ff", "KNN": "#ff9944", "RF": "#44ff88"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")
fig.suptitle("ROC Curves — All Models", color="white", fontsize=14, fontweight='bold')

for idx, name in enumerate(["SVM", "KNN", "RF"]):
    ax = axes[idx]
    ax.set_facecolor("#1a1f2e")
    fpr, tpr, roc_auc = roc_data[name]
    lc = ["#ff4444", "#ff9944", "#44ff88", "#44aaff"]
    ln = CLASS_NAMES
    for i in range(4):
        ax.plot(fpr[i], tpr[i], color=lc[i], lw=2,
                label=f"{ln[i]} (AUC={roc_auc[i]:.2f})")
    ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
    ax.set_title(f"{name}", color=colors_roc[name], fontsize=12, fontweight='bold')
    ax.set_xlabel("False Positive Rate", color="white", fontsize=9)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=9)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.legend(fontsize=7, facecolor='#0f1117', labelcolor='white')
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("../static/outputs/roc.png", bbox_inches='tight',
            facecolor='#0f1117', dpi=120)
plt.close()
print("📊 ROC curve saved")

# ── COMBINED CONFUSION MATRIX PLOT ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")
fig.suptitle("Confusion Matrices — All Models", color="white", fontsize=14, fontweight='bold')

for idx, name in enumerate(["SVM", "KNN", "RF"]):
    ax   = axes[idx]
    cm   = cm_data[name]
    acc  = results[name]
    im   = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f"{name}  ({acc*100:.1f}%)", color=colors_roc[name],
                 fontsize=12, fontweight='bold')
    ax.set_facecolor("#1a1f2e")
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks); ax.set_xticklabels(CLASS_NAMES, rotation=30,
                                                   ha='right', color='white', fontsize=8)
    ax.set_yticks(tick_marks); ax.set_yticklabels(CLASS_NAMES, color='white', fontsize=8)
    ax.set_xlabel("Predicted", color='white', fontsize=9)
    ax.set_ylabel("Actual",    color='white', fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i,j] < thresh else 'black', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("../static/outputs/confusion_matrices.png", bbox_inches='tight',
            facecolor='#0f1117', dpi=120)
plt.close()
print("📊 Confusion matrices saved")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
best_model = max(results, key=results.get)
print("\n" + "="*50)
print("🎉 ALL DONE!")
print("="*50)
for name, acc in results.items():
    print(f"  {name}: {acc*100:.2f}%")
print(f"\n🏆 BEST MODEL: {best_model} ({results[best_model]*100:.2f}%)")
print("="*50)
print("\n✅ Files saved to ml/:")
print("   brain_tumor_advanced_model.pkl, svm_meta_model.pkl, svm_woa_model.pkl")
print("   feature_scaler.pkl, feature_pca.pkl, feature_lda.pkl, feature_nca.pkl")
print("   eval_data.pkl")
print("\n✅ Restart Streamlit to load new models!")