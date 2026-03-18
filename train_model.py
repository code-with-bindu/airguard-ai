import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv('lung_risk_dataset.csv')

# Features and target
X = df[['aqi', 'age', 'smoking', 'outdoor_hours', 'exercise', 'condition']]
y = df['risk_level']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")