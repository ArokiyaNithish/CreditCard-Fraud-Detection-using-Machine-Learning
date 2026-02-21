# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# ===== File paths =====
MODEL_PATH = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/scripts/random_forest_model.pkl"
SCALER_PATH = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/scripts/scaler.pkl"

# ===== Load model and scaler =====
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    
    if request.method == "POST":
        try:
            # Get input values from form and convert to float
            features = [
                float(request.form["V1"]),
                float(request.form["V2"]),
                float(request.form["V3"]),
                float(request.form["V4"]),
                float(request.form["V5"]),
                float(request.form["V6"]),
                float(request.form["V7"]),
                float(request.form["V8"]),
                float(request.form["V9"]),
                float(request.form["V10"]),
                float(request.form["V11"]),
                float(request.form["V12"]),
                float(request.form["V13"]),
                float(request.form["V14"]),
                float(request.form["V15"]),
                float(request.form["V16"]),
                float(request.form["V17"]),
                float(request.form["V18"]),
                float(request.form["V19"]),
                float(request.form["V20"]),
                float(request.form["V21"]),
                float(request.form["V22"]),
                float(request.form["V23"]),
                float(request.form["V24"]),
                float(request.form["V25"]),
                float(request.form["V26"]),
                float(request.form["V27"]),
                float(request.form["V28"]),
                float(request.form["Amount"])
            ]

            # Convert to DataFrame for scaler
            input_df = pd.DataFrame([features], columns=[
                "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
                "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
                "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
            ])

            # Scale Amount only
            input_df['Amount_scaled'] = scaler.transform(input_df[['Amount']])
            input_df.drop(['Amount'], axis=1, inplace=True)

            # Predict
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                prediction_text = f"🚨 Fraud Detected! (Probability: {probability:.2%})"
            else:
                prediction_text = f"✅ Legitimate Transaction (Probability: {probability:.2%})"
        
        except ValueError:
            prediction_text = "❌ Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
