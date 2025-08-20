import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from preprocess import preprocess_img
import numpy as np

# ----------------
# Config
# ----------------
THRESHOLD = 0.5 # adjust if needed
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models
nails_model = load_model(os.path.join(MODELS_DIR, "nails_model.h5"))
conj_model  = load_model(os.path.join(MODELS_DIR, "conj_model.h5"))

# ----------------
# FastAPI setup
# ----------------
app = FastAPI(title="Anemia Detection API")

# Allow Flutter app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict to your teammate's IP/domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------
# Routes
# ----------------
@app.get("/")
def root():
    return {"ok": True, "msg": "Anemia API running"}

@app.post("/predict")
async def predict_anemia(
    fingernails: UploadFile = File(...),
    conjunctiva: UploadFile = File(...)
):
    # Read uploaded bytes
    nail_bytes = await fingernails.read()
    conj_bytes = await conjunctiva.read()

    # Preprocess directly from bytes
    nail_input = preprocess_img(nail_bytes)
    conj_input = preprocess_img(conj_bytes)

    # Predict
    nail_score = float(nails_model.predict(nail_input, verbose=0)[0][0])
    conj_score = float(conj_model.predict(conj_input, verbose=0)[0][0])
    nail_score = max(0.0, min(1.0, nail_score))
    conj_score = max(0.0, min(1.0, conj_score))

    combined_score = (nail_score + conj_score) / 2
    prediction = "Anemic" if combined_score >= THRESHOLD else "Non-Anemic"


    return {
        "nails_score": nail_score,
        "conj_score": conj_score,
        "combined_score": combined_score,
        "threshold": THRESHOLD,
        "prediction": prediction
    }

