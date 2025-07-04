import streamlit as st
import joblib
import pandas as pd
import requests
from urllib.parse import urlparse
from feature_extraction import extract_features
from datetime import datetime
import time
import base64
import os
from dotenv import load_dotenv
import csv
from pathlib import Path
import io
import socket
import glob
import smtplib
import schedule
import shap
import matplotlib.pyplot as plt
import threading
import time as t
from email.message import EmailMessage
from twilio.rest import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="wide")

# Load environment variables
load_dotenv()
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
SAFE_BROWSING_API_KEY = os.getenv("SAFE_BROWSING_API_KEY")

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ALERT_RECEIVER_EMAIL = os.getenv("ALERT_RECEIVER_EMAIL")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ALERT_RECEIVER_PHONE = os.getenv("ALERT_RECEIVER_PHONE")

if not VIRUSTOTAL_API_KEY or not SAFE_BROWSING_API_KEY:
    st.error("API keys missing! Please set VIRUSTOTAL_API_KEY and SAFE_BROWSING_API_KEY in your .env file.")
    st.stop()
LOG_FILE = Path("analysis_log.csv")
TRAIN_DATA_FILE = Path("data/processed/dataset_features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "phishing_url_detector.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

LOG_FILE = Path("analysis_log.csv")
REPORT_FILE = Path("reported_phishing.csv")
LOG_FILE = Path("analysis_log.csv")
TRAINING_DATA_FILE = Path("training_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load model and feature columns used during training
@st.cache_resource
def load_model():
    model = joblib.load('models/phishing_url_detector.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')  # Load saved feature columns list
    return model, feature_columns

# --- CSS Styling ---
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f7f9fc;
}
h1, h2, h3 {
    color: #0b3d91;
}
.phishing {
    color: white;
    background-color: #d32f2f;
    padding: 8px 14px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 20px;
    text-align: center;
}
.legit {
    color: white;
    background-color: #388e3c;
    padding: 8px 14px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 20px;
    text-align: center;
}
.stButton>button {
    background-color: #0b3d91;
    color: white;
    font-weight: bold;
    border-radius: 6px;
    height: 45px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Phishing Detection Quiz Data (you can expand this)
quiz_data = [
    {"url": "http://secure-login-apple.com", "label": 1, "explanation": "Fake Apple phishing URL using suspicious domain."},
    {"url": "https://www.google.com", "label": 0, "explanation": "Official Google website."},
    {"url": "http://paypal.verify-account-login.com", "label": 1, "explanation": "Phishing URL mimicking PayPal login."},
    {"url": "https://github.com", "label": 0, "explanation": "Trusted GitHub site."},
    {"url": "http://free-gift-cards.xyz", "label": 1, "explanation": "Suspicious TLD and offer, typical phishing."},
    {"url": "https://amazon.com", "label": 0, "explanation": "Official Amazon website."},
    {"url": "http://update-facebook-secure-login.com", "label": 1, "explanation": "Phishing attempt pretending to be Facebook login."},
]

def phishing_quiz():
    st.header("🕵️ Phishing Detection Quiz")
    st.write("Try to identify if the URL shown is Legitimate or Phishing. Good luck!")

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_question_num" not in st.session_state:
        st.session_state.quiz_question_num = 0
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    if st.session_state.quiz_question_num >= len(quiz_data):
        st.success(f"🎉 Quiz Completed! Your final score is {st.session_state.quiz_score} out of {len(quiz_data)}.")

        score_pct = st.session_state.quiz_score / len(quiz_data)
        if score_pct == 1.0:
            badge = "🏆 Phishing Master"
        elif score_pct >= 0.75:
            badge = "🥇 Cybersecurity Pro"
        elif score_pct >= 0.5:
            badge = "🥈 Security Enthusiast"
        else:
            badge = "🔰 Keep Learning!"

        st.markdown(f"**Your Badge:** {badge}")

        if st.button("Restart Quiz"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_question_num = 0
            st.session_state.quiz_submitted = False
        return

    current = quiz_data[st.session_state.quiz_question_num]
    st.write(f"**URL:** {current['url']}")

    if not st.session_state.quiz_submitted:
        user_answer = st.radio("Is this URL Legitimate or Phishing?", ("Legitimate", "Phishing"), key="quiz_answer")

        if st.button("Submit Answer"):
            correct_label = "Phishing" if current['label'] == 1 else "Legitimate"
            if user_answer == correct_label:
                st.success("Correct! 🎉")
                st.session_state.quiz_score += 1
            else:
                st.error(f"Wrong! The correct answer is **{correct_label}**.")
            st.info(current['explanation'])
            st.session_state.quiz_submitted = True
    else:
        st.info(current['explanation'])
        if st.button("Next Question"):
            st.session_state.quiz_question_num += 1
            st.session_state.quiz_submitted = False

def auto_train_model():
    if not TRAIN_DATA_FILE.exists():
        st.warning(" No training data found. Please accumulate some data first.")
        return None, None

    df = pd.read_csv(TRAIN_DATA_FILE)
    if df.empty:
        st.warning(" No data to train on.")
        return None, None
        

    # Add advanced features for better phishing detection
    df['has_suspicious_keywords'] = df['url'].str.contains(r"login|verify|secure|apple|bank|update", case=False).astype(int)
    df['uses_info_tld'] = df['url'].str.endswith(".info").astype(int)

    X = df.drop(columns=['url', 'label'])
    y = df['label']
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model retrained with accuracy: {acc:.2%}")

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    st.info("Model and feature columns saved.")

    return clf, feature_columns
def explain_prediction(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    st.subheader("🔍 Model Explainability")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig)

model, FEATURE_COLUMNS = load_model()

if model is None or FEATURE_COLUMNS is None:
    with st.spinner("Training model..."):
        model, FEATURE_COLUMNS = auto_train_model()

# 🧠 Auto-Retrain Background Job

def schedule_auto_retrain():
    def job():
        auto_train_model()

    schedule.every().day.at("02:00").do(job)

    def run_schedule():
        while True:
            schedule.run_pending()
            t.sleep(60)

    threading.Thread(target=run_schedule, daemon=True).start()

schedule_auto_retrain()

model, FEATURE_COLUMNS = load_model()

if model is None or FEATURE_COLUMNS is None:
    with st.spinner("Training model..."):
        model, FEATURE_COLUMNS = auto_train_model()

def get_ip_info(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc if parsed.netloc else parsed.path
        if not domain:
            return {"error": "No domain found in URL."}

        # Remove port if exists
        domain = domain.split(":")[0]

        # Resolve to IP
        ip_address = socket.gethostbyname(domain)

        response = requests.get(f"https://ipinfo.io/{ip_address}/json")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "details": response.text}
    except socket.gaierror:
        return {"error": f"Domain '{domain}' could not be resolved to IP."}
    except Exception as e:
        return {"error": str(e)}


def check_google_safe_browsing(url):
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={SAFE_BROWSING_API_KEY}"
    body = {
        "client": {"clientId": "phishing-detector-app", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    resp = requests.post(endpoint, json=body)
    if resp.status_code == 200:
        data = resp.json()
        return {"status": "dangerous", "details": data["matches"]} if "matches" in data else {"status": "clean"}
    return {"error": f"HTTP {resp.status_code}", "details": resp.text}


def check_virustotal(url):
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}
    try:
        # Submit URL for scanning
        resp_submit = requests.post("https://www.virustotal.com/api/v3/urls", headers=headers, data={"url": url})
        if resp_submit.status_code != 200:
            return {"error": f"Submit failed HTTP {resp_submit.status_code}", "details": resp_submit.text}
        analysis_id = resp_submit.json()["data"]["id"]

        # Poll analysis report until completed or timeout (~30s)
        for _ in range(15):
            time.sleep(2)
            resp_report = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}", headers=headers)
            if resp_report.status_code != 200:
                return {"error": f"Report fetch failed HTTP {resp_report.status_code}", "details": resp_report.text}
            result = resp_report.json()
            status = result.get("data", {}).get("attributes", {}).get("status")
            if status == "completed":
                return result
        return {"error": "Analysis timed out, try again later."}
    except Exception as e:
        return {"error": str(e)}


def display_vt_report(vt_result):
    if "error" in vt_result:
        st.error(f"VirusTotal Error: {vt_result['error']}")
        if "details" in vt_result:
            st.text(vt_result["details"])
        return

    stats = vt_result.get("data", {}).get("attributes", {}).get("stats", {})
    if stats:
        cols = st.columns(len(stats))
        for i, (k, v) in enumerate(stats.items()):
            cols[i].metric(label=k.capitalize(), value=str(v))

    st.markdown("---")
    st.write("Full VirusTotal JSON report:")
    st.json(vt_result)


def log_analysis(url, prediction, proba):
    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "URL", "Prediction", "Confidence_Legit", "Confidence_Phishing"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            url,
            "Phishing" if prediction == 1 else "Legitimate",
            f"{proba[0]:.4f}",
            f"{proba[1]:.4f}"
        ])

def save_reported_url(url, user_comment=""):
    file_exists = REPORT_FILE.exists()
    with open(REPORT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "URL", "User Comment"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            url,
            user_comment
        ])

def send_email_alert(url, prediction, confidence):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not ALERT_RECEIVER_EMAIL:
        st.warning("Email alert credentials not set!")
        return

    msg = EmailMessage()
    msg['Subject'] = f"⚠️ Phishing Alert: Suspicious URL Detected"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ALERT_RECEIVER_EMAIL
    body = f"""
Alert! A URL has been flagged as phishing.

URL: {url}
Prediction: {prediction}
Confidence (Phishing): {confidence:.2%}

Please review and take necessary action.
"""
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        st.success("Email alert sent!")
    except Exception as e:
        st.error(f"Failed to send email alert: {e}")


def send_sms_alert(url, prediction, confidence):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, ALERT_RECEIVER_PHONE]):
        st.warning("Twilio SMS alert credentials not set!")
        return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    body = (
        f"⚠️ Phishing Alert!\n"
        f"URL: {url}\n"
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence:.2%}"
    )

    try:
        message = client.messages.create(
            body=body,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_RECEIVER_PHONE
        )
        st.success("SMS alert sent!")
    except Exception as e:
        st.error(f"Failed to send SMS alert: {e}")


def main():
    st.title("🛡️ Intelligent Phishing URL Detector")

    # Sidebar options
    st.sidebar.header("Options")
    show_features = st.sidebar.checkbox("Show extracted features")
    show_confidence = st.sidebar.checkbox("Show prediction confidence")
    show_ipinfo = st.sidebar.checkbox("Show IP / Domain info")
    check_vt = st.sidebar.checkbox("Check VirusTotal")
    check_gsb = st.sidebar.checkbox("Check Google Safe Browsing")
    show_history = st.sidebar.checkbox("Show Analysis History")

    # Sidebar toggle for quiz
    quiz_mode = st.sidebar.checkbox("Take Phishing Detection Quiz")

    if quiz_mode:
        phishing_quiz()
        return

    model, FEATURE_COLUMNS = load_model()

    url_input = st.text_input("Enter URL to analyze")

    if st.button("Analyze"):
        if not url_input.strip():
            st.warning("Please enter a URL!")
            return


        with st.spinner("Extracting features and running ML model..."):
            features = extract_features(url_input)
            df = pd.DataFrame([features])
            # Reindex columns to match training features, fill missing with 0
            df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
            prediction = model.predict(df)[0]
            proba = model.predict_proba(df)[0]

        if prediction == 1:
            st.markdown('<div class="phishing">⚠️ Likely PHISHING link!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="legit">✅ URL appears LEGITIMATE.</div>', unsafe_allow_html=True)

        if show_confidence:
            st.info(f"Confidence Scores:\n- Legitimate: {proba[0]:.2%}\n- Phishing: {proba[1]:.2%}")

        if show_features:
            st.subheader("Extracted Features")
            st.json(features)

        # Log result
        log_analysis(url_input, prediction, proba)

        # Get IP info early for layout
        ip_info = get_ip_info(url_input) if show_ipinfo else None
        vt_result = None
        gsb_result = None

        if check_vt:
            st.subheader("🛡️ VirusTotal Report")
            with st.spinner("Querying VirusTotal... (may take ~20-30 seconds)"):
                vt_result = check_virustotal(url_input)
            display_vt_report(vt_result)

        if check_gsb:
            st.subheader("🔍 Google Safe Browsing Report")
            with st.spinner("Querying Google Safe Browsing..."):
                gsb_result = check_google_safe_browsing(url_input)
            st.json(gsb_result)

        # Layout IP info + VT + GSB side by side if more than one selected
        col_items = []
        if show_ipinfo:
            col_items.append(("🌐 IP / Domain Info", ip_info))
        if check_vt:
            col_items.append(("🛡️ VirusTotal Report JSON", vt_result))
        if check_gsb:
            col_items.append(("🔍 Google Safe Browsing JSON", gsb_result))

        if len(col_items) > 1:
            cols = st.columns(len(col_items))
            for idx, (title, content) in enumerate(col_items):
                with cols[idx]:
                    st.subheader(title)
                    if content:
                        st.json(content)
                    else:
                        st.info("No data.")
        elif len(col_items) == 1:
            st.subheader(col_items[0][0])
            if col_items[0][1]:
                st.json(col_items[0][1])
            else:
                st.info("No data.")

    # Batch URL analysis section
    st.markdown("---")
    st.header("Batch URL Analysis")

    uploaded_file = st.file_uploader("Upload CSV or TXT file with URLs (one per line or in 'url' column)", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_urls = pd.read_csv(uploaded_file)
                # Expecting a column named 'url' or first column as URLs
                if 'url' in df_urls.columns:
                    urls = df_urls['url'].dropna().unique().tolist()
                else:
                    # fallback: take first column as URLs
                    urls = df_urls.iloc[:, 0].dropna().unique().tolist()
            else:
                # TXT file, read line by line
                urls = [line.strip() for line in io.TextIOWrapper(uploaded_file) if line.strip()]

            st.info(f"Loaded {len(urls)} URLs for batch analysis.")

            results = []
            with st.spinner("Analyzing URLs... This may take some time."):
                for url in urls:
                    try:
                        features = extract_features(url)
                        df_features = pd.DataFrame([features])
                        df_features = df_features.reindex(columns=FEATURE_COLUMNS, fill_value=0)  # Reindex here too
                        pred = model.predict(df_features)[0]
                        proba = model.predict_proba(df_features)[0]
                        results.append({
                            "url": url,
                            "prediction": "Phishing" if pred == 1 else "Legitimate",
                            "confidence_legit": proba[0],
                            "confidence_phishing": proba[1]
                        })
                    except Exception as e:
                        results.append({
                            "url": url,
                            "prediction": "Error",
                            "confidence_legit": None,
                            "confidence_phishing": None,
                            "error": str(e)
                        })

            # Convert to DataFrame
            df_results = pd.DataFrame(results)

            # Show summary stats
            st.subheader("Batch Analysis Summary")
            phishing_count = (df_results['prediction'] == "Phishing").sum()
            legit_count = (df_results['prediction'] == "Legitimate").sum()
            error_count = (df_results['prediction'] == "Error").sum()

            st.write(f"✅ Legitimate URLs: {legit_count}")
            st.write(f"⚠️ Phishing URLs: {phishing_count}")
            if error_count > 0:
                st.write(f"❌ Errors: {error_count}")

            # Show results table
            st.subheader("Analysis Results")
            st.dataframe(df_results)

            # Provide download button for CSV export
            csv_export = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_export,
                file_name="batch_url_analysis_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Failed to process uploaded file: {str(e)}")

    # Phishing Report Submission
    st.markdown("---")
    st.header("🚨 Report a Suspicious URL")

    with st.form("report_form"):
        reported_url = st.text_input("Enter suspicious URL you want to report")
        user_comment = st.text_area("Additional info (optional)")
        submitted = st.form_submit_button("Submit Report")

    if submitted:
        if not reported_url.strip():
            st.warning("Please enter a URL to report.")
        else:
            save_reported_url(reported_url, user_comment)
            st.success("Thank you! The URL has been submitted for review.")

    # Show past analysis history
    if show_history:
        st.markdown("---")
        st.subheader("📜 Past URL Analysis Log")
        if LOG_FILE.exists():
            df_log = pd.read_csv(LOG_FILE)
            st.dataframe(df_log)
        else:
            st.info("No history found yet.")

    st.markdown("---")
    st.caption("Made with ❤️ for your cybersecurity project")


if __name__ == "__main__":
    main()
