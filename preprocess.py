import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


IMG_SIZE = (128, 128)

data = []
labels = []

label_map = {
    "anemic": 1,
    "non-anemic": 0
}

base_path = r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\raw"
output_base_dir = r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\processed"

for category in ["conjunctiva", "fingernails"]:
    category_path = os.path.join(base_path, category)
    for label_name, label_value in label_map.items():
        folder_path = os.path.join(category_path, label_name)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)  
                img = img / 255.0  
                data.append(img)
                labels.append(label_value)


data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

train_dir = os.path.join(output_base_dir, "train")
test_dir = os.path.join(output_base_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for idx, img in enumerate(X_train):
    cv2.imwrite(os.path.join(train_dir, f"train_{idx}.png"), img)

for idx, img in enumerate(X_test):
    cv2.imwrite(os.path.join(test_dir, f"test_{idx}.png"), img)

print(f"Images saved to:\n{train_dir}\n{test_dir}")
