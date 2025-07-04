# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os
from feature_extraction import extract_features

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_DATA = "data/processed/dataset_cleaned.csv"
FEATURE_DATA = "data/processed/dataset_features.csv"

def generate_feature_dataframe(input_csv):
    df = pd.read_csv(input_csv)
    features = []

    for url in df['url']:
        try:
            feats = extract_features(url)
            features.append(feats)
        except Exception as e:
            print(f"❌ Error extracting features from {url}: {e}")
            continue

    feature_df = pd.DataFrame(features)
    feature_df['label'] = df['label'][:len(feature_df)]
    return feature_df

def train_model():
    df = generate_feature_dataframe(INPUT_DATA)

    # Save extracted features
    df.to_csv(FEATURE_DATA, index=False)
    print(f"✅ Feature dataset saved to {FEATURE_DATA}")

    # Drop rows with NaN in the 'label' column
    df.dropna(subset=['label'], inplace=True)

    X = df.drop(columns=['label'])
    y = df['label']
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and feature columns
    joblib.dump(model, os.path.join(MODEL_DIR, "phishing_url_detector.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print("✅ Model and feature columns saved in 'models/'")

if __name__ == "__main__":
    train_model()