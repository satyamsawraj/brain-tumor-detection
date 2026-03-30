import os

base = os.path.dirname(os.path.abspath(__file__))
print("Root:", base)

paths_to_check = [
    "ml/brain_tumor_advanced_model.pkl",
    "ml/svm_meta_model.pkl", 
    "ml/svm_woa_model.pkl",
    "ml/brain_tumor_advanced_model.pkl",
    "ml/svm_meta_model.pkl",
]

for p in paths_to_check:
    full = os.path.join(base, p)
    print(f"{'✅' if os.path.exists(full) else '❌'} {p}")
