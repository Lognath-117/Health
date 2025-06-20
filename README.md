# 🧠 SymptomAtlas — AI-Powered Early Multi-Disease Detection

This is a fully deployable standalone AI web app using Streamlit.

## 🚀 Features

- Detect early multi-disease risks from behavioral data.
- Supports both synthetic demo dataset & user-uploaded CSV.
- Full ML pipeline (classification, drift analysis, explainability).
- Built with production-grade Streamlit design.
- SHAP explainability integrated.
- Professional UI for both technical & non-technical users.

## 🗂️ Files

- `app.py` — Main Streamlit app.
- `requirements.txt` — Python dependencies.

## 🔧 Deployment (Streamlit Cloud)

1️⃣ Fork this repository to your own GitHub.  
2️⃣ Connect GitHub repo to [Streamlit Cloud](https://streamlit.io/cloud).  
3️⃣ Deploy directly with no code change.

## ⚠️ Kaggle Data

If you have a Kaggle dataset, modify `app.py` (function: `download_kaggle_data()`) and include your Kaggle credentials (`kaggle.json`) in deployment environment.

## 🧪 Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
