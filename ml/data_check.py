import os

DATASET_PATH = "../dataset"

splits = ["Training", "Testing"]
classes = ["glioma", "meningioma", "pituitary", "notumor"]

for split in splits:
    print(f"\n📂 {split} Data:")
    for cls in classes:
        folder = os.path.join(DATASET_PATH, split, cls)
        if os.path.exists(folder):
            count = len(os.listdir(folder))
            print(f"  {cls}: {count} images")
        else:
            print(f"  {cls}: folder not found")
