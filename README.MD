# 🫁 AirGuard AI — Lung Risk Predictor

> An AI-powered web app that predicts lung health risk based on air quality and lifestyle data.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow?style=flat-square)

---

## 🔴 Live Demo
👉 [Try it on Hugging Face](https://huggingface.co/spaces/YOUR_USERNAME/airguard-ai)

---

## 📌 What It Does

AirGuard AI takes 6 inputs from the user and predicts their lung health risk as **Low**, **Moderate**, or **High** — along with confidence score and personalized recommendations.

| Input | Description |
|---|---|
| AQI Level | City's Air Quality Index (0–500) |
| Age | User's age |
| Daily Outdoor Hours | Time spent outside |
| Smoking Status | Non-smoker to Heavy |
| Exercise Frequency | Rarely to Daily |
| Respiratory Condition | None to COPD |

---

## 🧠 ML Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Data | 50,000 synthetic records |
| Features | 6 (AQI, age, smoking, outdoor hours, exercise, condition) |
| Test Accuracy | 91.4% |
| Output | 3 classes — Low / Moderate / High Risk |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Python, scikit-learn, pandas, numpy |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Hugging Face Spaces (Docker) |

---

## 📁 Project Structure

```
airguard-ai/
├── app.py              # FastAPI backend + API routes
├── index.html          # Frontend UI
├── generate_data.py    # Synthetic dataset generation
├── train_model.py      # Model training script
├── model.pkl           # Trained Random Forest model
├── requirements.txt    # Python dependencies
└── Dockerfile          # For Hugging Face deployment
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/airguard-ai.git
cd airguard-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data and train model
python generate_data.py
python train_model.py

# 4. Start the server
uvicorn app:app --reload

# 5. Open in browser
http://localhost:8000
```

---

## 📊 Sample Prediction

| Input | Value |
|---|---|
| AQI | 280 |
| Age | 55 |
| Smoking | Heavy |
| Outdoor Hours | 6h |
| Exercise | Rarely |
| Condition | COPD |
| **Result** | 🔴 **High Risk — 94.2% confidence** |

---

## 👩‍💻 Built By

**Bindu** — AI/ML Enthusiast · LPU Student  
[GitHub](https://github.com/YOUR_USERNAME) · [Hugging Face](https://huggingface.co/YOUR_USERNAME)

---

> ⚠️ This project is for educational and demo purposes only. Not a substitute for medical advice.