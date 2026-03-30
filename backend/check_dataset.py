import os

DATASET_DIR = "../dataset/Training"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

for cls in CLASSES:
    count = len(os.listdir(os.path.join(DATASET_DIR, cls)))
    print(f"{cls}: {count} images")
