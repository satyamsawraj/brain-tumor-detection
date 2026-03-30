import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load your saved features (use same as previous step)
X = np.load("X_train_all.npy")
y = np.load("y_train_mc.npy")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Metaheuristic Optimization (Random Search)
best_acc = 0
best_params = {}

print("🔍 Searching best parameters...")

for i in range(20):  # iterations
    C = np.random.uniform(0.1, 100)
    gamma = np.random.uniform(0.001, 1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Iter {i+1}: C={C:.2f}, gamma={gamma:.4f}, acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_params = {"C": C, "gamma": gamma}

print("\n✅ BEST RESULT")
print("Best Accuracy:", best_acc)
print("Best Params:", best_params)

# Train final model with best params
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"], probability=True))
])

final_model.fit(X_train, y_train)

joblib.dump(final_model, "svm_meta_model.pkl")
print("💾 Metaheuristic optimized model saved!")