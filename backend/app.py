from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import librosa
import tensorflow_hub as hub
import io  # FIXED: use Python's io instead of librosa.io

# ---------------------------
# LOAD MODELS
# ---------------------------
ml_model = joblib.load("models/best_ml_model.pkl")
scaler = joblib.load("models/ml_scaler.pkl")
label_classes = np.load("models/ml_label_encoder_classes.npy", allow_pickle=True)

# Load YAMNet from TF Hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# ---------------------------
# FASTAPI APP SETUP
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {"message": "Audio Spoof Detection API Running"}


# ---------------------------
# FEATURE EXTRACTION (YAMNET)
# ---------------------------
def extract_yamnet_embedding(audio):
    scores, embeddings, _ = yamnet_model(audio)
    return embeddings.numpy().mean(axis=0)


# ---------------------------
# REAL / SPOOF MAPPING
# ---------------------------
def map_to_real_or_spoof(label):
    if label == "0PR":
        return "REAL"
    else:
        return "SPOOF"


# ---------------------------
# PREDICT ROUTE
# ---------------------------
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # FIXED: librosa.io DOES NOT EXIST
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    # Pad short audio
    if len(y) < 15600:
        y = np.pad(y, (0, 15600 - len(y)))

    y = y.astype(np.float32)

    # Extract YAMNet embeddings
    embedding = extract_yamnet_embedding(y)

    # Scale embedding
    scaled = scaler.transform([embedding])

    # Predict class
    probs = ml_model.predict_proba(scaled)[0]
    pred_idx = np.argmax(probs)

    raw_label = str(label_classes[pred_idx])
    readable_label = map_to_real_or_spoof(raw_label)

    return {
        "prediction": readable_label,      # REAL or SPOOF
        "raw_label": raw_label,            # 0PR / 1PR / 2PR (for debugging)
        "confidence": float(probs[pred_idx])
    }
