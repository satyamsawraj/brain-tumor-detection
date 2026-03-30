import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
import gdown

def download_models():
    os.makedirs("ml", exist_ok=True)

    links = {
        "brain_tumor_advanced_model.pkl": "https://drive.google.com/uc?id=1juA7MN94QhAYbkg24ktz3lk7VrHxBbba",
        "brain_tumor_final_model.pkl": "https://drive.google.com/uc?id=1JzRz7Wpx6WQS51b6luiXwUYABTUrvJod",
        "brain_tumor_multiclass_model.pkl": "https://drive.google.com/uc?id=1IcL--fpY6HcLdITrJoQrWjcwM9I9fVmg",
        "brain_tumor_svm_model.pkl": "https://drive.google.com/uc?id=1JKwHe1SzVPc75geVZ5XbDQGWgpcJMEVK",
        "ensemble_model.pkl": "https://drive.google.com/uc?id=1HcdW19TjQpeZdI1PBBRx0lT4laddpt6J",
        "eval_data.pkl": "https://drive.google.com/uc?id=1cjqkDNBVDH8paSDHi63LJxoQG_yZAI5C",
        "feature_lda.pkl": "https://drive.google.com/uc?id=1fm779TqP4IQVSCqErmQG2wIRZVbXcB4L",
        "feature_nca.pkl": "https://drive.google.com/uc?id=1Q4Ha457ese4lNdmoymgZR7glJojNAZzs",
        "feature_pca.pkl": "https://drive.google.com/uc?id=1URbKz6AWEYi2aoUEdXsOuXip602ssZiU",
        "feature_scaler.pkl": "https://drive.google.com/uc?id=1f2v2zCfRkimZ_rWwYziwsOQCy79Xnj1V",
        "knn_model.pkl": "https://drive.google.com/uc?id=1RmsgR5gWX86uEQvTpdG9w-rAlFxwOjtv",
        "mc_ensemble.pkl": "https://drive.google.com/uc?id=1vGv7kvE3hwjaV1m6Wb0jhm-Yla9h4tBg",
        "mc_knn.pkl": "https://drive.google.com/uc?id=1L4YHvj44B67A4UOOaLbk1I81b0l_iJop",
        "mc_rf.pkl": "https://drive.google.com/uc?id=1PXMvg9MKPl_2LYvwyfF1fXJNGk5lULIz",
        "mc_svm.pkl": "https://drive.google.com/uc?id=1-8qS93cVRIwalg-ml-XRj09lzFheFLlC",
        "pca.pkl": "https://drive.google.com/uc?id=15oZh9wD8zpyTTeltT9CT6FOer7P6EgsN",
        "rf_model.pkl": "https://drive.google.com/uc?id=1X9OhNRxpdIkz5gvA9m_sZaLNlUH_LEfi",
        "svm_meta_model.pkl": "https://drive.google.com/uc?id=1UBdgcVJPoNGgZxHhiFcxdKDzhzCHnd3z",
        "svm_model.pkl": "https://drive.google.com/uc?id=1lEhNsfn0WOHGkmJSZ0qxDNDiYI2W7DG9",
        "svm_woa_model.pkl": "https://drive.google.com/uc?id=1wrTvRyzv9Dn8jYLBm0qLpiLSdcA373mX"
    }

    for filename, url in links.items():
        path = os.path.join("ml", filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            gdown.download(url, path, quiet=False)

download_models()

import streamlit as st
import numpy as np
import cv2
import joblib
import sys
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import time
from collections import Counter

# ── PATH SETUP ──────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ml")))
from advanced_features import extract_features

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .hero-banner {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #00d4ff33; border-radius: 16px;
        padding: 2rem 2.5rem; margin-bottom: 1.5rem; text-align: center;
    }
    .hero-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff6b6b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .hero-subtitle { color: #8892a4; font-size: 1rem; margin-top: 0.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #00d4ff22; border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
    .prediction-box { border-radius: 14px; padding: 1.5rem 2rem; text-align: center; margin: 1rem 0; }
    .pred-tumor { background: linear-gradient(135deg, #3d1a1a, #5c2020); border: 2px solid #ff4444; }
    .pred-no-tumor { background: linear-gradient(135deg, #1a3d1a, #205c20); border: 2px solid #44ff44; }
    .pred-label { font-size: 1.8rem; font-weight: 800; margin: 0; }
    .pred-confidence { font-size: 1rem; opacity: 0.8; margin-top: 0.3rem; }
    .model-badge {
        background: #1a1f2e; border: 1px solid #00d4ff33;
        border-radius: 8px; padding: 0.6rem 1rem; text-align: center; margin: 0.3rem 0;
    }
    .model-name { color: #8892a4; font-size: 0.75rem; }
    .model-pred { color: #ffffff; font-size: 1rem; font-weight: 600; }
    .section-header {
        color: #00d4ff; font-size: 1.1rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        border-bottom: 1px solid #00d4ff33; padding-bottom: 0.5rem; margin-bottom: 1rem;
    }
    .warning-box {
        background: #2d2000; border: 1px solid #ffaa00;
        border-radius: 10px; padding: 1rem; color: #ffaa00; font-size: 0.85rem; margin-top: 1rem;
    }
    .sidebar-stat {
        background: #1a1f2e; border-radius: 8px;
        padding: 0.8rem; margin: 0.4rem 0; border-left: 3px solid #00d4ff;
    }
    .history-item {
        background: #1a1f2e; border-radius: 8px; padding: 0.7rem 1rem;
        margin: 0.3rem 0; border-left: 3px solid #7b2fff; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = {"Glioma": "#ff4444", "Meningioma": "#ff9944",
                "No Tumor": "#44ff88", "Pituitary": "#44aaff"}
CLASS_EMOJI  = {"Glioma": "🔴", "Meningioma": "🟠", "No Tumor": "🟢", "Pituitary": "🔵"}
CLASS_INFO   = {
    "Glioma":     "Gliomas arise from glial cells. They're the most common primary brain tumors.",
    "Meningioma": "Meningiomas grow from the meninges (brain membranes). Usually benign.",
    "No Tumor":   "No tumor detected. Brain tissue appears within normal parameters.",
    "Pituitary":  "Pituitary tumors form in the pituitary gland. Often treatable."
}
MODEL_COLORS = {"SVM": "#00d4ff", "KNN": "#ff9944", "RF": "#44ff88"}
IMG_SIZE = 224

# ── LOAD EVERYTHING ──────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    base = os.path.join(os.path.dirname(__file__), "ml")

    # Models
    models = {}
    for key, fname in [("advanced", "brain_tumor_advanced_model.pkl"),
                       ("meta",     "svm_meta_model.pkl"),
                       ("woa",      "svm_woa_model.pkl")]:
        try:
            models[key] = joblib.load(os.path.join(base, fname))
        except Exception as e:
            models[key] = None

    # Transformers
    transformers = {}
    for key, fname in [("scaler", "feature_scaler.pkl"), ("pca", "feature_pca.pkl"),
                       ("lda",    "feature_lda.pkl"),    ("nca", "feature_nca.pkl")]:
        try:
            transformers[key] = joblib.load(os.path.join(base, fname))
        except:
            transformers[key] = None

    # Eval data (ROC + CM)
    try:
        eval_data = joblib.load(os.path.join(base, "eval_data.pkl"))
    except:
        eval_data = None

    return models, transformers, eval_data

# ── PREPROCESSING ────────────────────────────────────────────────────────────
clahe_proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return clahe_proc.apply(gray)

# ── FEATURE EXTRACTION ───────────────────────────────────────────────────────
def get_features(processed, transformers):
    f = extract_features(processed, use_ltp=True)
    if isinstance(f, tuple): f = f[0]
    f = np.array(f, dtype=np.float64).reshape(1, -1)
    if all(transformers.get(k) is not None for k in ["scaler","pca","lda","nca"]):
        try:
            f_scaled = transformers["scaler"].transform(f)
            f_pca    = transformers["pca"].transform(f_scaled)
            f_lda    = transformers["lda"].transform(f_scaled)
            f_nca    = transformers["nca"].transform(f_scaled)
            f        = np.hstack([f_pca, f_lda, f_nca])
        except Exception as e:
            print(f"⚠️ Transformer failed: {e}")
    return f

# ── PREDICTION ───────────────────────────────────────────────────────────────
def run_prediction(processed, models, transformers):
    features = get_features(processed, transformers)
    results  = {}
    for key in ["advanced", "meta", "woa"]:
        if not models.get(key): continue
        try:
            proba = models[key].predict_proba(features)[0]
            pred  = int(np.argmax(proba))
            results[key] = {
                "prediction":    CLASS_NAMES[pred],
                "confidence":    float(np.max(proba)) * 100,
                "probabilities": {CLASS_NAMES[i]: float(p)*100 for i,p in enumerate(proba)},
            }
        except Exception as e:
            results[key] = {"error": str(e)}
    votes = [r["prediction"] for r in results.values() if "prediction" in r]
    if votes:
        final    = Counter(votes).most_common(1)[0][0]
        avg_conf = float(np.mean([r["confidence"] for r in results.values() if "confidence" in r]))
    else:
        final, avg_conf = "Unknown", 0.0
    return results, final, avg_conf

# ── ROC FIGURE ───────────────────────────────────────────────────────────────
def make_roc_fig(eval_data):
    roc_data = eval_data["roc"]
    lc = ["#ff4444", "#ff9944", "#44ff88", "#44aaff"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")
    fig.suptitle("ROC Curves — Per Class per Model", color="white", fontsize=13, fontweight='bold')

    for idx, name in enumerate(["SVM", "KNN", "RF"]):
        ax = axes[idx]
        ax.set_facecolor("#1a1f2e")
        fpr, tpr, roc_auc = roc_data[name]
        for i in range(4):
            ax.plot(fpr[i], tpr[i], color=lc[i], lw=2,
                    label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:.2f})")
        ax.plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
        ax.set_title(name, color=MODEL_COLORS[name], fontsize=12, fontweight='bold')
        ax.set_xlabel("False Positive Rate", color="white", fontsize=9)
        ax.set_ylabel("True Positive Rate",  color="white", fontsize=9)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        ax.legend(fontsize=7, facecolor='#0f1117', labelcolor='white', loc='lower right')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig

# ── CONFUSION MATRIX FIGURE ──────────────────────────────────────────────────
def make_cm_fig(eval_data):
    cm_data  = eval_data["cm"]
    accuracy = eval_data["accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")
    fig.suptitle("Confusion Matrices — All Models", color="white", fontsize=13, fontweight='bold')

    for idx, name in enumerate(["SVM", "KNN", "RF"]):
        ax  = axes[idx]
        cm  = cm_data[name]
        acc = accuracy[name]
        im  = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_facecolor("#1a1f2e")
        ax.set_title(f"{name}   Accuracy: {acc*100:.1f}%",
                     color=MODEL_COLORS[name], fontsize=11, fontweight='bold')
        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', color='white', fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASS_NAMES, color='white', fontsize=8)
        ax.set_xlabel("Predicted", color='white', fontsize=9)
        ax.set_ylabel("Actual",    color='white', fontsize=9)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=11,
                        color='white' if cm[i,j] < thresh else 'black', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    return fig

# ── FEATURE VIZ ──────────────────────────────────────────────────────────────
def make_dtcwt_fig(img, label):
    try:
        import dtcwt
        transform = dtcwt.Transform2d()
        coeffs = transform.forward(img.astype(np.float32), nlevels=3)
        fig = plt.figure(figsize=(8, 6), facecolor="#0f1117")
        fig.suptitle(f"DTCWT Decomposition — {label}", color="white", fontsize=11, fontweight='bold')
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
        for i, title in enumerate(["Original", "Level 1", "Level 2", "Level 3"]):
            ax = fig.add_subplot(gs[i//2, i%2])
            ax.imshow(img if i==0 else np.abs(coeffs.highpasses[i-1][:,:,0]),
                      cmap="gray" if i==0 else "inferno")
            ax.set_title(title, color="#00d4ff", fontsize=9); ax.axis("off")
        return fig
    except:
        fig, ax = plt.subplots(facecolor="#0f1117")
        ax.text(0.5,0.5,"dtcwt not available",color="white",ha='center',va='center')
        ax.axis("off"); return fig

def make_gabor_fig(img, label):
    fig = plt.figure(figsize=(10, 6), facecolor="#0f1117")
    fig.suptitle(f"Gabor Filter Analysis — {label}", color="white", fontsize=11, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.2)
    panels = [(img,"gray","Original"),
              *[(cv2.filter2D(img,cv2.CV_32F,
                 cv2.getGaborKernel((21,21),5,np.radians(a),10,0.5,0)),
                 "hot",f"{a}° Gabor") for a in [0,45,90,135]],
              (cv2.Canny(img,50,150),"gray","Edge Detection")]
    for i,(data,cmap,title) in enumerate(panels):
        ax = fig.add_subplot(gs[i//3,i%3])
        ax.imshow(data,cmap=cmap)
        ax.set_title(title,color="#00d4ff",fontsize=9); ax.axis("off")
    return fig

def make_entropy_fig(img, label):
    try:
        from skimage.filters.rank import entropy as rank_entropy
        from skimage.morphology import disk
        emap = rank_entropy(img, disk(5))
    except:
        emap = np.zeros_like(img)
    fig, axes = plt.subplots(1, 2, figsize=(9,4), facecolor="#0f1117")
    fig.suptitle(f"Entropy Analysis — {label}", color="white", fontsize=11, fontweight='bold')
    axes[0].imshow(img,cmap="gray"); axes[0].set_title("Original",color="#00d4ff",fontsize=9); axes[0].axis("off")
    im = axes[1].imshow(emap,cmap="jet"); axes[1].set_title("Local Entropy Map",color="#00d4ff",fontsize=9); axes[1].axis("off")
    cbar = fig.colorbar(im,ax=axes[1],fraction=0.046)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color='white')
    plt.tight_layout(); return fig

def make_prob_fig(probabilities):
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [CLASS_COLORS[l] for l in labels]
    fig, ax = plt.subplots(figsize=(7,3.5), facecolor="#0f1117")
    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(min(val+1,98), bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%', va='center', color='white', fontsize=10, fontweight='bold')
    ax.set_xlim(0,105); ax.set_facecolor("#0f1117")
    ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False); ax.set_title("Class Probabilities",color="white",fontsize=10)
    fig.tight_layout(); return fig

# ── SESSION STATE ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0

models, transformers, eval_data = load_all()
transformers_ready = all(transformers.get(k) is not None for k in ["scaler","pca","lda","nca"])

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 BrainScan AI")
    st.markdown("---")
    st.markdown("**Model Status**")
    model_labels = {"advanced": "Advanced SVM", "meta": "Meta (KNN)", "woa": "WOA (RF)"}
    for key, label in model_labels.items():
        status = "🟢" if models.get(key) else "🔴"
        acc_str = ""
        if eval_data and eval_data["accuracy"]:
            ak = {"advanced":"SVM","meta":"KNN","woa":"RF"}[key]
            acc_str = f"{eval_data['accuracy'].get(ak,0)*100:.1f}%"
        st.markdown(f"""
        <div class="sidebar-stat">
            {status} <b>{label}</b><br>
            <span style="color:#8892a4;font-size:0.78rem">Accuracy: {acc_str}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Transformer Pipeline**")
    t_color  = "#44ff88" if transformers_ready else "#ffaa00"
    t_status = "🟢 PCA+LDA+NCA Active" if transformers_ready else "🟡 Raw 28 features"
    st.markdown(f'<div class="sidebar-stat"><span style="color:{t_color}">{t_status}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Session Stats**")
    st.markdown(f'<div class="sidebar-stat">📊 Total Scans: <b>{st.session_state.total_scans}</b></div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Settings**")
    show_viz   = st.toggle("Feature Visualizations", value=True)
    show_proba = st.toggle("Probability Chart",       value=True)
    show_all   = st.toggle("All Model Results",       value=True)
    st.markdown("---")
    st.markdown("**About**")
    st.caption("SVM+KNN+RF ensemble with PCA+LDA+NCA reduction. For research use only.")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("**Recent History**")
        for h in reversed(st.session_state.history[-5:]):
            color = CLASS_COLORS.get(h['pred'],'#ffffff')
            st.markdown(f"""
            <div class="history-item">
                <span style="color:{color}">● {h['pred']}</span>
                <span style="color:#8892a4;float:right">{h['conf']:.0f}%</span><br>
                <span style="color:#555;font-size:0.75rem">{h['time']}</span>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🧠 Brain Tumor Detection AI</p>
    <p class="hero-subtitle">SVM + KNN + RF Ensemble • PCA + LDA + NCA Reduction • DTCWT + Gabor Features</p>
</div>
""", unsafe_allow_html=True)

# Top metrics
c1, c2, c3, c4 = st.columns(4)
best_acc = max(eval_data["accuracy"].values())*100 if eval_data else 0
for col, val, label in [
    (c1, "3",            "Active Models"),
    (c2, f"{best_acc:.1f}%", "Best Accuracy"),
    (c3, "PCA+LDA+NCA",  "Feature Pipeline"),
    (c4, "4",            "Tumor Classes")
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.4rem">{val}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_predict, tab_roc, tab_cm = st.tabs([
    "🔬 MRI Prediction",
    "📈 ROC Curves",
    "🔢 Confusion Matrices"
])

# ════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════
with tab_predict:
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown('<div class="section-header">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose MRI image", type=["jpg","jpeg","png","bmp","tiff"],
                                    label_visibility="collapsed")
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_pil    = Image.open(io.BytesIO(uploaded.getvalue()))
            st.image(img_pil, caption=f"📁 {uploaded.name}", use_container_width=True)
            st.markdown(f"""
            <div style="background:#1a1f2e;border-radius:8px;padding:0.7rem;font-size:0.82rem;color:#8892a4;margin-top:0.5rem">
                📄 <b style="color:white">{uploaded.name}</b><br>
                📐 {img_pil.size[0]}×{img_pil.size[1]} px &nbsp;|&nbsp; 💾 {uploaded.size/1024:.1f} KB
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("🔬 Analyze MRI", type="primary", use_container_width=True)
            st.markdown("""
            <div class="warning-box">⚠️ <b>Research Use Only</b> — Not a substitute for professional medical diagnosis.</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1a1f2e;border:2px dashed #00d4ff33;border-radius:12px;
                        padding:3rem;text-align:center;color:#8892a4">
                🖼️<br><br>Drag & drop your MRI scan here<br>
                <span style="font-size:0.8rem">Supports JPG, PNG, BMP, TIFF</span>
            </div>""", unsafe_allow_html=True)
            analyze_btn = False

    with right:
        st.markdown('<div class="section-header">🎯 Analysis Results</div>', unsafe_allow_html=True)

        if uploaded and analyze_btn:
            prog = st.progress(0, text="Preprocessing image...")
            time.sleep(0.3)
            processed = preprocess(img_bgr)
            prog.progress(20, text="Extracting raw features (Gabor + LTP)...")
            time.sleep(0.3)
            prog.progress(45, text="Applying PCA + LDA + NCA reduction...")
            time.sleep(0.3)
            prog.progress(65, text="Running SVM + KNN + RF ensemble...")
            results, final_pred, avg_conf = run_prediction(processed, models, transformers)
            prog.progress(90, text="Generating report...")
            time.sleep(0.2)
            prog.progress(100, text="Complete!")
            time.sleep(0.3)
            prog.empty()

            st.session_state.total_scans += 1
            st.session_state.history.append({
                "pred": final_pred, "conf": avg_conf,
                "time": datetime.now().strftime("%H:%M:%S"), "file": uploaded.name
            })

            is_tumor  = final_pred != "No Tumor"
            box_class = "pred-tumor" if is_tumor else "pred-no-tumor"
            emoji     = CLASS_EMOJI.get(final_pred, "❓")
            color     = CLASS_COLORS.get(final_pred, "#ffffff")

            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <p class="pred-label" style="color:{color}">{emoji} {final_pred}</p>
                <p class="pred-confidence">Ensemble Confidence: <b>{avg_conf:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)

            st.info(f"ℹ️ {CLASS_INFO.get(final_pred,'')}")

            if show_proba:
                adv = results.get("advanced", {})
                if "probabilities" in adv:
                    st.markdown("**Class Probability Distribution**")
                    st.pyplot(make_prob_fig(adv["probabilities"]), use_container_width=True)

            if show_all:
                st.markdown("**Individual Model Predictions**")
                mc1, mc2, mc3 = st.columns(3)
                for col, key, name in [(mc1,"advanced","Advanced SVM"),
                                       (mc2,"meta","Meta (KNN)"),
                                       (mc3,"woa","WOA (RF)")]:
                    r = results.get(key, {})
                    if "prediction" in r:
                        c = CLASS_COLORS.get(r["prediction"],"#ffffff")
                        col.markdown(f"""
                        <div class="model-badge">
                            <div class="model-name">{name}</div>
                            <div class="model-pred" style="color:{c}">{r['prediction']}</div>
                            <div style="color:#8892a4;font-size:0.78rem">{r['confidence']:.1f}% conf</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        col.markdown(f"""
                        <div class="model-badge">
                            <div class="model-name">{name}</div>
                            <div class="model-pred" style="color:#ff4444">Error</div>
                            <div style="color:#555;font-size:0.7rem">{str(r.get('error',''))[:40]}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="color:#555;font-size:0.78rem;text-align:right;margin-top:1rem">
                🕐 Analyzed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>""", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div style="background:#1a1f2e;border-radius:12px;padding:3rem;text-align:center;color:#8892a4">
                <br>📋<br><br>Upload an MRI scan and click
                <b style="color:#00d4ff">Analyze MRI</b> to see results here<br><br>
            </div>""", unsafe_allow_html=True)

    # Feature visualizations
    if uploaded and analyze_btn and show_viz and 'processed' in dir():
        st.markdown("---")
        st.markdown('<div class="section-header">🔬 Feature Visualizations</div>', unsafe_allow_html=True)
        vt1, vt2, vt3 = st.tabs(["🌀 DTCWT", "🔥 Gabor Filters", "🌡️ Entropy"])
        with vt1:
            st.markdown("Dual-Tree Complex Wavelet Transform captures multi-scale frequency features.")
            st.pyplot(make_dtcwt_fig(processed, final_pred), use_container_width=True)
        with vt2:
            st.markdown("Log-Gabor filters with attention weighting extract oriented texture features.")
            st.pyplot(make_gabor_fig(processed, final_pred), use_container_width=True)
        with vt3:
            st.markdown("Local entropy analysis reveals texture complexity and information content.")
            st.pyplot(make_entropy_fig(processed, final_pred), use_container_width=True)

# ════════════════════════════════════════════════
# TAB 2 — ROC CURVES
# ════════════════════════════════════════════════
with tab_roc:
    st.markdown('<div class="section-header">📈 ROC Curves — SVM vs KNN vs RF</div>',
                unsafe_allow_html=True)
    if eval_data and eval_data.get("roc"):
        st.markdown("Each chart shows per-class ROC curves. Higher AUC = better discrimination.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(make_roc_fig(eval_data), use_container_width=True)

        # AUC summary table
        st.markdown("---")
        st.markdown("**AUC Summary Table**")
        import pandas as pd
        auc_rows = []
        for name in ["SVM","KNN","RF"]:
            _, _, roc_auc = eval_data["roc"][name]
            row = {"Model": name}
            for i, cls in enumerate(CLASS_NAMES):
                row[cls] = f"{roc_auc[i]:.3f}"
            row["Mean AUC"] = f"{np.mean([roc_auc[i] for i in range(4)]):.3f}"
            auc_rows.append(row)
        auc_df = pd.DataFrame(auc_rows).set_index("Model")
        st.dataframe(auc_df, use_container_width=True)
    else:
        st.warning("⚠️ ROC data not found. Please run `retrain_all.py` first to generate `eval_data.pkl`.")

# ════════════════════════════════════════════════
# TAB 3 — CONFUSION MATRICES
# ════════════════════════════════════════════════
with tab_cm:
    st.markdown('<div class="section-header">🔢 Confusion Matrices — SVM vs KNN vs RF</div>',
                unsafe_allow_html=True)
    if eval_data and eval_data.get("cm"):
        st.markdown("Rows = Actual class, Columns = Predicted class. Diagonal = correct predictions.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(make_cm_fig(eval_data), use_container_width=True)

        # Accuracy summary
        st.markdown("---")
        st.markdown("**Model Accuracy Summary**")
        import pandas as pd
        acc_rows = [{"Model": name, "Accuracy": f"{acc*100:.2f}%",
                     "Correct": int(eval_data["cm"][name].trace()),
                     "Total":   int(eval_data["cm"][name].sum())}
                    for name, acc in eval_data["accuracy"].items()]
        acc_df = pd.DataFrame(acc_rows).set_index("Model")
        st.dataframe(acc_df, use_container_width=True)
    else:
        st.warning("⚠️ Confusion matrix data not found. Please run `retrain_all.py` first to generate `eval_data.pkl`.")

# ── SCAN HISTORY ─────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Scan History</div>', unsafe_allow_html=True)
    import pandas as pd
    df = pd.DataFrame(st.session_state.history)
    df.columns = ["Prediction","Confidence (%)","Time","File"]
    df["Confidence (%)"] = df["Confidence (%)"].round(1)
    df.index = range(1, len(df)+1)
    st.dataframe(df, use_container_width=True, height=200)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.total_scans = 0
        st.rerun()

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:0.8rem;padding:1rem">
    🧠 Brain Tumor Detection AI &nbsp;|&nbsp; Research Interface &nbsp;|&nbsp;
    SVM+KNN+RF Ensemble • PCA+LDA+NCA Pipeline • DTCWT+Gabor+LTP Features<br>
    <span style="color:#ff4444">⚕️ Not for clinical use. For research and educational purposes only.</span>
</div>""", unsafe_allow_html=True)