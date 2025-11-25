# Audio Spoof Detector

This project detects whether an audio file is REAL or SPOOF using ML.

## Tech Stack
- FastAPI backend
- HTML/CSS/JS frontend
- TensorFlow + YAMNet for audio embeddings
- Scikit-learn ML model (SVM / best-model.pkl)
- Uvicorn server

## How to run
cd backend  
pip install -r requirements.txt  
uvicorn app:app --reload --port 8000

Open frontend/index.html in browser.
