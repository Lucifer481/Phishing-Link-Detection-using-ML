"""
app.py
Flask app for phishing link detection.
"""

from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from feature_engineer import extract_features

app = Flask(__name__)
model = joblib.load("models/phishing_rf_model.joblib")

HTML_PAGE = """
<!doctype html>
<title>Phishing Link Detector</title>
<h2>Phishing Link Detection (Commercial Bank)</h2>
<form method=post>
  URL: <input type=text name=url size=60>
  <input type=submit value=Check>
</form>
{% if result %}
  <h3>Prediction: {{ result }}</h3>
  <p>Confidence: Legit={{ legit }} | Phishing={{ phishing }}</p>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    legit = phishing = 0.0
    if request.method == 'POST':
        url = request.form['url']
        features = extract_features(url)
        df = pd.DataFrame([features])
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        result = "PHISHING" if pred == 1 else "LEGITIMATE"
        legit = f"{proba[0]:.2f}"
        phishing = f"{proba[1]:.2f}"
    return render_template_string(HTML_PAGE, result=result, legit=legit, phishing=phishing)

if __name__ == '__main__':
    app.run(debug=True)
