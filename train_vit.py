import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
THRESHOLD = 0.5
FINE_TUNE_LAYERS = 20  
  
# Paths
nails_train_folder = "data/processed/fingernails/train"
nails_test_folder  = "data/processed/fingernails/test"
conj_train_folder  = "data/processed/conjunctiva/train"
conj_test_folder   = "data/processed/conjunctiva/test"

# ----------------------------
# DATA GENERATORS
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

nails_train_gen = train_datagen.flow_from_directory(
    nails_train_folder, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
)
nails_test_gen = test_datagen.flow_from_directory(
    nails_test_folder, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

conj_train_gen = train_datagen.flow_from_directory(
    conj_train_folder, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=True
)
conj_test_gen = test_datagen.flow_from_directory(
    conj_test_folder, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# ----------------------------
# CHECK CLASS INDICES
# ----------------------------
print("Nails class indices:", nails_train_gen.class_indices)
print("Conjunctiva class indices:", conj_train_gen.class_indices)

# If anemic=0 in your dataset, we can flip prediction later
flip_nails = nails_train_gen.class_indices.get('anemic', 1) == 0
flip_conj  = conj_train_gen.class_indices.get('anemic', 1) == 0

# ----------------------------
# CLASS WEIGHTS
# ----------------------------
def get_class_weights(generator):
    classes = generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    return dict(enumerate(class_weights))

nails_class_weights = get_class_weights(nails_train_gen)
conj_class_weights = get_class_weights(conj_train_gen)

# ----------------------------
# CREATE TRANSFER LEARNING MODEL WITH FINE-TUNING
# ----------------------------
def create_transfer_model(fine_tune_layers=FINE_TUNE_LAYERS):
    base_model = MobileNetV2(input_shape=(128,128,3), include_top=False, weights='imagenet')
    base_model.trainable = True

    # Freeze all layers except the last `fine_tune_layers`
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# TRAINING FUNCTION
# ----------------------------
def train_model(train_gen, test_gen, class_weights, save_path):
    model = create_transfer_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    ]
    model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, callbacks=callbacks, class_weight=class_weights)
    return model

# ----------------------------
# TRAIN BOTH MODELS
# ----------------------------
print("Training nails model...")
nails_model = train_model(nails_train_gen, nails_test_gen, nails_class_weights, "nails_mobilenet_finetune.h5")

print("Training conjunctiva model...")
conj_model = train_model(conj_train_gen, conj_test_gen, conj_class_weights, "conj_mobilenet_finetune.h5")

print("Training complete. Models saved.")

nails_model.save("backend/models/nails_model.h5")
conj_model.save("backend/models/conj_model.h5")


# ----------------------------
# INFERENCE / PREDICTION
# ----------------------------
def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_anemia(nails_img_path, conj_img_path, threshold=THRESHOLD):
    nails_pred = nails_model.predict(preprocess_img(nails_img_path))[0][0]
    conj_pred  = conj_model.predict(preprocess_img(conj_img_path))[0][0]

    # Handle class index flip
    if flip_nails:
        nails_pred = 1 - nails_pred
    if flip_conj:
        conj_pred = 1 - conj_pred

    # ----- DEBUG STEP -----
    print(f"Debug: Nails prediction = {nails_pred:.3f}, Conjunctiva prediction = {conj_pred:.3f}")

    # Combine predictions by averaging
    combined = (nails_pred + conj_pred)/2
    return "Anemic" if combined >= threshold else "Non-Anemic"

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
example_nails_img = r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\processed\fingernails\test\non-anemic\fingernails_non-anemic_25.png"
example_conj_img  = r"C:\Users\narma\OneDrive\Desktop\anemia-hack\data\processed\conjunctiva\test\non-anemic\conjunctiva_non-anemic_1.png"
result = predict_anemia(example_nails_img, example_conj_img)
print(f"Combined prediction: {result}")

