import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

aqi        = np.random.randint(0, 500, n)
age        = np.random.randint(5, 90, n)
smoking    = np.random.randint(0, 4, n)   # 0=none, 1=occasional, 2=regular, 3=heavy
outdoor    = np.round(np.random.uniform(0, 12, n), 1)
exercise   = np.random.randint(0, 4, n)   # 0=rarely, 1=1-2x, 2=3-5x, 3=daily
condition  = np.random.randint(0, 4, n)   # 0=none, 1=asthma, 2=bronchitis, 3=COPD

# Risk score calculation based on real medical research weights
score = (
    (aqi / 500) * 35
    + np.where(age < 18, 0.3, np.where(age < 40, 0.4, np.where(age < 60, 0.7, 1.0))) * 15
    + (smoking / 3) * 25
    + (outdoor / 12) * np.where(aqi > 150, 10, 4)
    - (exercise / 3) * 10
    + (condition / 3) * 20
)

score = np.clip(score, 0, 100)

# Add realistic noise to make it a real ML problem
score += np.random.normal(0, 3, n)
score = np.clip(score, 0, 100)

# Convert score to risk category (what model will predict)
risk = np.where(score < 30, 0, np.where(score < 60, 1, 2))
# 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk

df = pd.DataFrame({
    'aqi': aqi,
    'age': age,
    'smoking': smoking,
    'outdoor_hours': outdoor,
    'exercise': exercise,
    'condition': condition,
    'risk_level': risk
})

df.to_csv('lung_risk_dataset.csv', index=False)
print(f"Dataset saved! Shape: {df.shape}")
print(df['risk_level'].value_counts().sort_index())
print(df.head())