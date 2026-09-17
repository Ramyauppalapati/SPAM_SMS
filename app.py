from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from ml_utils import ensure_model

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered",
)

MODEL_PATH = Path("models/spam_classifier.joblib")

@st.cache_resource(show_spinner="Preparing the spam detector model...")
def get_model():
    return ensure_model()

try:
    model = get_model()
except Exception as exc:
    st.error("The model could not be prepared.")
    st.exception(exc)
    st.stop()

st.title("📩 SMS Spam Detection")
st.caption("NLP + Machine Learning • TF-IDF + Logistic Regression")

st.subheader("Check a message")
message = st.text_area(
    "Enter an SMS message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize...",
)

if st.button("Detect Spam", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        spam_probability = float(probabilities[classes.index("spam")])

        if prediction == "spam":
            st.error(f"🚨 SPAM — confidence: {spam_probability:.1%}")
        else:
            st.success(f"✅ HAM (not spam) — confidence: {(1-spam_probability):.1%}")

st.divider()

st.subheader("Try an example")
col1, col2 = st.columns(2)

with col1:
    if st.button("Example: spam"):
        st.session_state["example"] = (
            "Congratulations! You have won a free prize. Call now to claim your reward!"
        )

with col2:
    if st.button("Example: normal"):
        st.session_state["example"] = (
            "Hey, are we still meeting for lunch today?"
        )

if "example" in st.session_state:
    st.info(st.session_state["example"])

st.subheader("Model information")
st.markdown("""
- **Text features:** word + character TF-IDF n-grams
- **Classifier:** Logistic Regression
- **Evaluation:** accuracy, precision, recall, F1-score
- **Deployment:** Streamlit
""")

metrics_path = Path("reports/metrics.csv")
if metrics_path.exists():
    st.subheader("Evaluation metrics")
    metrics = pd.read_csv(metrics_path)
    metrics["score"] = metrics["score"].map(lambda x: f"{x:.4f}")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

st.caption("Educational/demo project. Do not use automated classification as the sole basis for consequential decisions.")
