import cv2
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224
DATASET_PATH = "../dataset/Training/glioma"

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# pick one sample image
img_name = os.listdir(DATASET_PATH)[0]
img_path = os.path.join(DATASET_PATH, img_name)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

clahe_img = clahe.apply(img)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original MRI")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("CLAHE Enhanced")
plt.imshow(clahe_img, cmap='gray')
plt.axis('off')

plt.show()
