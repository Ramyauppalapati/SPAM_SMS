# 📩 SMS Spam Detection — Python, ML, NLP

An end-to-end SMS spam classification project using NLP and machine learning, with an interactive Streamlit web application.

## 🚀 Live deployment

This repository is designed to deploy directly to **Streamlit Community Cloud** from GitHub.

The app can automatically download the public SMS Spam Collection dataset and train the model if `models/spam_classifier.joblib` is not present. This makes the repository deployable without committing the dataset.

## ✨ Features

- Binary SMS classification: **ham vs spam**
- Word-level TF-IDF features
- Character-level TF-IDF features
- Logistic Regression classifier
- Class-weight balancing
- Accuracy, precision, recall and F1-score
- Confusion matrix
- Class-distribution visualization
- Interactive Streamlit UI
- Automatic model preparation on first deployment
- Command-line prediction support

## 📁 Project structure

```text
sms-spam-detection/
├── app.py
├── ml_utils.py
├── train.py
├── predict.py
├── download_dataset.py
├── requirements.txt
├── runtime.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── raw/
│       └── .gitkeep
├── models/
│   └── spam_classifier.joblib   # optional; generated locally or on deployment
└── reports/
    ├── figures/
    │   ├── class_distribution.png
    │   └── confusion_matrix.png
    └── generated metrics files
```

## 💻 Run locally in VS Code

Open the repository folder in VS Code and run:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python download_dataset.py
```

Train and evaluate:

```bash
python train.py
```

Start the app:

```bash
streamlit run app.py
```

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app from your GitHub repository.
4. Select the `main` branch.
5. Set the main file to:

```text
app.py
```

6. Deploy.

The app will install packages from `requirements.txt`. If the trained model is not committed, `app.py` automatically downloads the dataset and trains the model on first startup.

### Recommended GitHub repository

```text
sms-spam-detection
```

### Recommended Streamlit settings

```text
Branch: main
Main file: app.py
Python: 3.12
```

## 🧠 Machine-learning pipeline

```text
SMS message
     ↓
TF-IDF word n-grams
     +
TF-IDF character n-grams
     ↓
Feature Union
     ↓
Logistic Regression
     ↓
HAM / SPAM
```

The model uses word bigrams to capture phrases and character n-grams to improve robustness to spelling variations and short/obfuscated text.

## 📊 Evaluation

Run:

```bash
python train.py
```

Generated files:

```text
reports/metrics.csv
reports/classification_report.txt
reports/figures/class_distribution.png
reports/figures/confusion_matrix.png
```

Exact scores depend on the installed scikit-learn version and the train/test split.

## 🧪 Command-line prediction

After installing dependencies:

```bash
python predict.py "Congratulations! You won a free prize. Call now!"
```

Example output:

```text
Prediction: SPAM
Spam probability: 99.XX%
```

## 📦 Dataset

The project uses the **SMS Spam Collection** dataset from the UCI Machine Learning Repository.

The repository does not redistribute the dataset. `download_dataset.py` downloads it at runtime when needed.

Before publishing or redistributing this project, review the dataset's original terms and citation requirements.

## 📝 Resume-ready description

**SMS Spam Detection — Python, ML, NLP**
- Developed an SMS Spam Detection system using NLP and machine learning for accurate binary text classification.
- Built a TF-IDF pipeline using word and character n-grams with Logistic Regression.
- Deployed an interactive Streamlit web app enabling real-time spam detection.
- Evaluated models using accuracy, precision, recall and F1-score, with confusion-matrix and dataset-distribution visualizations.

## ⚠️ Disclaimer

This is an educational/demo machine-learning project. Predictions can be incorrect and should not be used as the sole basis for consequential decisions.
