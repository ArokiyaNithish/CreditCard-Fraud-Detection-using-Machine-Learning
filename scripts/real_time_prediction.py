# scripts/real_time_prediction.py

import pandas as pd
import joblib
import numpy as np
import os

# ============================
# Step 1: Load Model & Scaler
# ============================
model_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/scripts/random_forest_model.pkl"
scaler_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/scripts/scaler.pkl"

# Check if files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}. Please train the model first!")

if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"❌ Scaler file not found at {scaler_path}. Please run step3_preprocessing.py first!")

# Load model and scaler
rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("✅ Model and Scaler loaded successfully!")

# ============================
# Step 2: Example Transaction
# ============================
# Format: [V1, V2, ..., V28, Amount]
new_transaction = [[-1.359807, -0.072781, 2.536346, 1.378155, -0.338321, 0.462388,
                    0.239599, 0.098698, 0.363787, 0.090795, -0.551600, -0.617801,
                    -0.991390, -0.311169, 1.468177, -0.470400, 0.207971, 0.025791,
                    0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928,
                    0.128539, -0.189115, 0.133558, -0.021053, 149.62]]

# ============================
# Step 3: Convert to DataFrame
# ============================
columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
transaction_df = pd.DataFrame(new_transaction, columns=columns)

# ============================
# Step 4: Scale and Rename 'Amount'
# ============================
transaction_df['Amount'] = scaler.transform(transaction_df[['Amount']])

# Rename column to match training dataset
transaction_df.rename(columns={'Amount': 'Amount_scaled'}, inplace=True)

# ============================
# Step 5: Make Prediction
# ============================
prediction = rf_model.predict(transaction_df)
probability = rf_model.predict_proba(transaction_df)[0][1]  # Probability of being fraud

# ============================
# Step 6: Output Result
# ============================
if prediction[0] == 1:
    print("⚠️ FRAUD DETECTED: This transaction is likely fraudulent.")
    print(f"Fraud Probability: {probability:.2%}")
else:
    print("✅ LEGITIMATE TRANSACTION: This transaction seems safe.")
    print(f"Fraud Probability: {probability:.2%}")
