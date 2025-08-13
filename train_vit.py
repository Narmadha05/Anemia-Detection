# train_vit.py
import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import keras_cv
import matplotlib.pyplot as plt

IMG_SIZE = (128, 128)

def load_images_from_folder(folder_path):
    images = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            # Adjust this if your saved filenames have labels
            label = 1 if "anemic" in file else 0
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_images_from_folder(r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\processed\train")
X_test, y_test = load_images_from_folder(r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\processed\test")

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

input_shape = (128, 128, 3)

vit = keras_cv.models.VisionTransformer(
    include_top=False,
    image_size=128,
    patch_size=16,
    num_layers=8,
    hidden_dim=64,
    mlp_dim=128,
    num_heads=4,
    dropout=0.1,
    classifier_activation=None
)

inputs = keras.Input(shape=input_shape)
x = vit(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

model.save("anemia_vit_model.h5")
print("Model saved as anemia_vit_model.h5")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.legend()

plt.show()
