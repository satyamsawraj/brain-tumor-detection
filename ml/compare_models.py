import matplotlib.pyplot as plt

# =========================
# MODEL ACCURACIES (UPDATE IF NEEDED)
# =========================
models = [
    "SVM",
    "Metaheuristic SVM",
    "WOA-SVM"
]

accuracies = [
    0.91,   # Your normal SVM
    0.89,   # Metaheuristic
    0.72    # WOA (replace if updated)
]

# =========================
# PLOT GRAPH
# =========================
plt.figure(figsize=(8, 5))

bars = plt.bar(models, accuracies)

# Add values on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f"{height:.2f}", ha='center', va='bottom')

plt.title("Model Comparison for Brain Tumor Detection")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()