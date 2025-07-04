import joblib
from feature_extraction import extract_features
import pandas as pd

def predict_url(url):
    # Load the saved model
    model = joblib.load('models/phishing_url_detector.pkl')

    # Extract features from the input URL
    features = extract_features(url)
    df = pd.DataFrame([features])

    # Predict using the model
    prediction = model.predict(df)[0]

    return prediction

if __name__ == "__main__":
    print("Enter URL to check (type 'exit' to quit):")
    while True:
        url = input("URL> ").strip()
        if url.lower() == 'exit':
            break
        result = predict_url(url)
        if result == 1:
            print("⚠️ Warning: This URL is likely a PHISHING link!")
        else:
            print("✅ This URL appears to be LEGITIMATE.")
