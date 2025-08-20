import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = (128, 128)
label_map = {"anemic": 1, "non-anemic": 0}

base_path = r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\raw"
output_base_dir = r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\processed"

# Process each category separately: fingernails and conjunctiva
for category in ["conjunctiva", "fingernails"]:
    category_path = os.path.join(base_path, category)

    # Prepare output folders
    train_dir = os.path.join(output_base_dir, category, "train")
    test_dir  = os.path.join(output_base_dir, category, "test")
    for subfolder in ["anemic", "non-anemic"]:
        os.makedirs(os.path.join(train_dir, subfolder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subfolder), exist_ok=True)

    # Loop through labels
    for label_name, label_value in label_map.items():
        folder_path = os.path.join(category_path, label_name)
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        files = os.listdir(folder_path)
        # Split into train and test
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

        # Save train images
        for idx, file in enumerate(train_files):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                img = (img / 255.0 * 255).astype(np.uint8)  # keep as uint8 for saving
                cv2.imwrite(os.path.join(train_dir, label_name, f"{category}_{label_name}_{idx}.png"), img)

        # Save test images
        for idx, file in enumerate(test_files):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                img = (img / 255.0 * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(test_dir, label_name, f"{category}_{label_name}_{idx}.png"), img)

print("Preprocessing complete. Folders ready for two-branch model.")
