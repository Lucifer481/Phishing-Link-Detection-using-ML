from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from feature_extraction import extract_features
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for local testing (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

# Load model and features
model = joblib.load("models/phishing_url_detector.pkl")
FEATURE_COLUMNS = joblib.load("models/feature_columns.pkl")

@app.post("/predict")
async def predict_url(data: URLRequest):
    features = extract_features(data.url)
    df = pd.DataFrame([features])

    # Ensure correct feature order
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    return {
        "url": data.url,
        "prediction": "phishing" if prediction == 1 else "legitimate",
        "confidence": {
            "legitimate": proba[0],
            "phishing": proba[1]
        }
    }
