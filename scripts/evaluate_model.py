# evaluate_model.py
# Updated for cleaned dataset and new Random Forest model

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import sys

# ===== Step 1: File Paths =====
BASE_DIR = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project"
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "scripts", "random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scripts", "scaler.pkl")

# ===== Step 2: Check files =====
for file_path, name in zip([DATA_PATH, MODEL_PATH, SCALER_PATH],
                           ["Dataset", "Model", "Scaler"]):
    if not os.path.exists(file_path):
        print(f"❌ {name} file not found at {file_path}")
        sys.exit()

print("✅ All necessary files found!")

# ===== Step 3: Load Dataset =====
print("🔹 Loading cleaned dataset...")
data = pd.read_csv(DATA_PATH)
print(f"Dataset loaded successfully. Total records: {len(data)}")
print("Fraud vs Legitimate counts:\n", data['Class'].value_counts())

# ===== Step 4: Load Model & Scaler =====
print("🔹 Loading trained model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Model and Scaler loaded successfully!")

# ===== Step 5: Preprocess Data =====
# Amount is already scaled in cleaned dataset
if 'Amount' in data.columns:
    data.drop(['Amount'], axis=1, inplace=True)

X = data.drop('Class', axis=1)
y = data['Class']

# ===== Step 6: Split into Train and Test =====
print("🔹 Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# ===== Step 7: Make Predictions =====
print("🔹 Making predictions on test data...")
y_pred = model.predict(X_test)

# ===== Step 8: Evaluate the Model =====
print("\n===== Model Evaluation =====")

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 3. Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

print("\n🎯 Evaluation complete. Focus on precision, recall, and F1-score for Class 1 (Fraud).")
