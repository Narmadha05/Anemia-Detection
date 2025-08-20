from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

IMG_SIZE = (128, 128)

def preprocess_img(img):
    """
    Preprocess image for inference.
    img: either a file path (str) or bytes (from FastAPI UploadFile)
    Returns: numpy array of shape (1, 128, 128, 3)
    """
    if isinstance(img, bytes):
        # FastAPI upload
        img = Image.open(BytesIO(img)).convert("RGB")
        img = img.resize(IMG_SIZE)
    else:
        # Local file path
        img = image.load_img(img, target_size=IMG_SIZE)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

