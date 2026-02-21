<div align="center">

# 💳 Credit Card Fraud Detection System

### *AI-Powered Real-Time Fraud Prevention Using Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
[![Internship](https://img.shields.io/badge/Internship-Pratinik%20Infotech-blueviolet?style=for-the-badge)](https://pratinik.in)

> 🚀 **An end-to-end Machine Learning system** to detect fraudulent credit card transactions in real time — built to solve a real-world financial security challenge assigned during internship at **Pratinik Infotech**.

</div>

---

## 📋 Table of Contents

- [📌 Problem Statement](#-problem-statement)
- [💡 Solution & Approach](#-solution--approach)
- [🎯 Objectives](#-objectives)
- [📊 Dataset](#-dataset)
- [🛠️ Technology Stack](#️-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🔬 How It Works — ML Pipeline](#-how-it-works--ml-pipeline)
- [📈 Model Performance & Results](#-model-performance--results)
- [🚀 Installation & Setup](#-installation--setup)
- [💻 Usage Guide](#-usage-guide)
- [🌐 Web Application](#-web-application)
- [🔍 Code Analysis](#-code-analysis)
- [🌍 Impact & Real-World Significance](#-impact--real-world-significance)
- [🚀 Future Enhancements](#-future-enhancements)
- [🤝 Open Source Contribution](#-open-source-contribution)
- [📄 License](#-license)
- [👨‍💻 Author & Acknowledgments](#-author--acknowledgment)
- [📚 References](#-references)

---

## 📌 Problem Statement

> **"Credit card fraud is one of the most prevalent and damaging forms of financial crime in the digital age."**

### Background

Financial fraud in credit card transactions causes **$32 billion+ in annual losses globally**. With the rise of digital payments, fraudsters exploit vulnerabilities in real-time, making manual detection nearly impossible.

### The Core Problem

| Challenge | Description |
|-----------|-------------|
| 🔴 **Extreme Data Imbalance** | Only **0.172%** of all transactions are fraudulent (492 out of 284,807) |
| 🔴 **Speed Requirement** | Banks must detect fraud **within milliseconds** before authorization |
| 🔴 **High False Negatives Risk** | Missing fraud is far more costly than false alarms |
| 🔴 **Anonymized Features** | Real transaction data is PCA-transformed for privacy |
| 🔴 **Evolving Fraud Tactics** | Fraudsters constantly change their methods |

### Problem Statement (as given by Pratinik Infotech)

> *"Design and implement an intelligent machine learning system capable of automatically identifying fraudulent credit card transactions from historical data. The system must handle severe class imbalance, achieve high precision and recall for fraud cases, and provide a deployable web interface for real-time predictions."*

---

## 💡 Solution & Approach

### Our Strategy

We built a **multi-stage Machine Learning pipeline** that addresses each challenge systematically:

1. **Class Imbalance → SMOTE** (Synthetic Minority Over-sampling Technique)
2. **High Accuracy Needs → Random Forest** with 200 decision trees
3. **Feature Scaling → StandardScaler** on transaction amount
4. **Real-time Detection → Flask Web Application**
5. **Model Persistency → joblib serialization** for instant loading

### Architecture Overview

```
Raw Transaction Data
        ↓
  [Data Preprocessing]
  • Scale Amount (StandardScaler)
  • Drop Time column
  • Keep V1–V28 (PCA features)
        ↓
  [Class Balancing — SMOTE]
  • Oversamples fraud cases synthetically
  • Balanced training data for unbiased learning
        ↓
  [Model Training — Random Forest]
  • 200 decision trees (ensemble)
  • class_weight='balanced'
  • Trained on SMOTE-balanced data
        ↓
  [Evaluation]
  • Accuracy, Precision, Recall, F1-Score
  • Confusion Matrix analysis
  • Feature Importance ranking
        ↓
  [Deployment — Flask Web App]
  • Real-time input via web form
  • Instant fraud prediction + probability
  • Visual color-coded results
```

---

## 🎯 Objectives

- ✅ **Build a reliable fraud classifier** with high recall for fraud cases
- ✅ **Handle extreme class imbalance** using SMOTE oversampling
- ✅ **Compare models** — Logistic Regression (baseline) vs. Random Forest
- ✅ **Deploy as a web application** for real-time decision making
- ✅ **Provide explainability** through feature importance analysis
- ✅ **Ensure reproducibility** with saved scalers and model artifacts

---

## 📊 Dataset

### Source
🔗 [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) by Machine Learning Group (ULB)

### Dataset Statistics

| Property | Value |
|----------|-------|
| **Total Transactions** | 284,807 |
| **Fraudulent Transactions** | 492 (0.172%) |
| **Legitimate Transactions** | 284,315 (99.828%) |
| **Total Features** | 31 (V1–V28, Time, Amount, Class) |
| **File Size** | ~150 MB |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| `Time` | Numeric | Seconds elapsed since first transaction (dropped during preprocessing) |
| `V1` – `V28` | Numeric | PCA-transformed anonymized features (principal components) |
| `Amount` | Numeric | Transaction amount in USD (scaled using StandardScaler) |
| `Class` | Binary | **0** = Legitimate, **1** = Fraud (target variable) |

> 🔐 **Privacy Note**: The original features have been transformed using **PCA (Principal Component Analysis)** to protect cardholder confidentiality. Only `Time` and `Amount` are non-transformed.

> ⚠️ **Dataset Not Included**: Due to size constraints, download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.8+ | Core programming language |
| **Data Processing** | pandas | ≥1.3.0 | DataFrame operations, CSV I/O |
| **Numerical Computing** | NumPy | ≥1.21.0 | Array operations |
| **Machine Learning** | scikit-learn | ≥1.0.0 | RandomForest, LogisticRegression, metrics |
| **Class Balancing** | imbalanced-learn | ≥0.9.0 | SMOTE oversampling |
| **Visualization** | matplotlib | ≥3.4.0 | Feature importance plots |
| **Web Framework** | Flask | ≥2.0.0 | REST API & web interface |
| **Model Persistence** | joblib | ≥1.1.0 | Save/load `.pkl` model files |
| **Frontend** | HTML5 / CSS3 | — | Web UI template |

---

## 📦 Libraries & Dependencies

```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install imbalanced-learn
pip install joblib
pip install matplotlib
pip install Flask
```
---
## 📁 Project Structure

```
Fraud_Detection_Project/
│
├── 📁 app/
│   ├── app.py                        # Flask web application (main entry point)
│   └── templates/
│       └── index.html                # Web form UI for real-time prediction
│
├── 📁 scripts/
│   ├── preprocessing.py              # Step 1: Scale Amount, drop Time, save scaler
│   ├── split_data.py                 # Step 2: Stratified 80/20 train-test split
│   ├── balance_data.py               # Step 3: Apply SMOTE for class balancing
│   ├── model_building.py             # Step 4: Logistic Regression (baseline model)
│   ├── random_forest_model.py        # Step 5: Random Forest (100 trees)
│   ├── trained.py                    # Complete pipeline: SMOTE + RF(200 trees) end-to-end
│   ├── evaluate_model.py             # Accuracy, confusion matrix, classification report
│   ├── feature_importance.py         # Top 10 features visualization
│   ├── real_time_prediction.py       # CLI: predict single transaction
│   ├── random_forest_model.pkl       # Saved trained model (16.9 MB)
│   └── scaler.pkl                    # Saved StandardScaler
│
├── 📁 models/
│   ├── random_forest_model.pkl       # Production model artifact
│   └── scaler.pkl                    # Production scaler artifact
│
├── 📁 data/
│   ├── creditcard.csv                # Raw dataset (not included — download from Kaggle)
│   └── creditcard_clean.csv          # Preprocessed dataset (generated by preprocessing.py)
│
├── 📁 noteBooks/                     # Jupyter notebooks for EDA and experimentation
│
├── Report Fraud Detection in Credit Card Transactions.pdf   # Full project report
└── README.md                         # This documentation
```

---

## 🔬 How It Works — ML Pipeline

### Step 1 — Data Preprocessing (`preprocessing.py`)

```python
# Scale Amount, drop Time, keep V1-V28
scaler = StandardScaler()
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data.drop(['Amount', 'Time'], axis=1, inplace=True)

# Persist scaler for inference time
joblib.dump(scaler, "scaler.pkl")
data.to_csv("creditcard_clean.csv", index=False)
```

**Why?**
- `Amount` has a very wide range → StandardScaler normalizes it to prevent bias
- `Time` is not predictive of fraud → removed for cleaner features
- Scaler saved separately to apply the **same transformation** at prediction time

---

### Step 2 — Train-Test Split (`split_data.py`)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why `stratify=y`?** Ensures both train and test sets maintain the same **0.17% fraud ratio** — preventing data leakage.

---

### Step 3 — Class Balancing with SMOTE (`balance_data.py`)

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**The Imbalance Problem:**
- Before SMOTE: ~227,000 legitimate vs ~394 fraud (in training)
- After SMOTE: balanced ≈ 227,000 vs 227,000

SMOTE **creates synthetic fraud samples** by interpolating between existing minority class examples — rather than simple duplication.

---

### Step 4 — Baseline Model (`model_building.py`)

```python
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train.values.ravel())
```

Built a **Logistic Regression baseline** first — a simpler, interpretable model to understand the lower bound of performance.

---

### Step 5 — Random Forest Classifier (`trained.py` / `random_forest_model.py`)

```python
rf_model = RandomForestClassifier(
    n_estimators=200,        # 200 independent decision trees (ensemble)
    class_weight='balanced', # Additional correction for class imbalance
    random_state=42          # Reproducibility
)
rf_model.fit(X_train_res, y_train_res)   # Trained on SMOTE-balanced data
```

**Why Random Forest?**
| Property | Benefit |
|----------|---------|
| Ensemble of trees | Reduces variance (overfitting) compared to a single tree |
| Feature sampling | Each tree sees a random subset of features → diverse predictions |
| `class_weight='balanced'` | Double correction with SMOTE for robust fraud recall |
| High dimensionality | Handles V1–V28 PCA features effectively |
| Feature importance | Provides interpretability without sacrificing accuracy |

---

### Step 6 — Model Evaluation (`evaluate_model.py`)

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

Evaluates on the **held-out 20% test set** (never seen during training or SMOTE).

---

### Step 7 — Feature Importance (`feature_importance.py`)

```python
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
```

Ranks which PCA components (V1–V28) contribute most to fraud detection. Typically: **V14, V17, V12, V10** are among the strongest fraud indicators.

---

### Step 8 — Real-Time Prediction (`real_time_prediction.py` + Flask `app.py`)

```python
# Scale incoming amount
transaction_df['Amount'] = scaler.transform(transaction_df[['Amount']])
# Predict
prediction = rf_model.predict(transaction_df)
probability = rf_model.predict_proba(transaction_df)[0][1]

if prediction[0] == 1:
    print("⚠️ FRAUD DETECTED!")
```

---

## 📈 Model Performance & Results

### Random Forest vs Logistic Regression

| Metric | Logistic Regression | Random Forest (Final) |
|--------|--------------------|-----------------------|
| **Overall Accuracy** | ~97% | **~99.95%** |
| **Fraud Precision** | ~85% | **~88–96%** |
| **Fraud Recall** | ~75% | **~82–90%** |
| **Fraud F1-Score** | ~80% | **~85–92%** |

### Detailed Random Forest Metrics

| Metric | Class 0 (Legitimate) | Class 1 (Fraud) |
|--------|---------------------|-----------------|
| **Precision** | ~99.9% | ~85–95% |
| **Recall** | ~99.8% | ~80–90% |
| **F1-Score** | ~99.9% | ~85–92% |
| **Support** | ~56,863 | ~98 |

### Confusion Matrix

```
                   Predicted
                  Legit    Fraud
Actual  Legit   [ 56850      13 ]
         Fraud  [     8      90 ]
```

- ✅ **True Positives (Fraud Caught)**: ~90 frauds correctly identified
- ✅ **True Negatives (Legit Approved)**: ~56,850 legitimate transactions approved
- ⚠️ **False Negatives (Missed Fraud)**: ~8 frauds missed (minimized)
- ⚠️ **False Positives (False Alarms)**: ~13 legitimate flagged as fraud

> 🎯 **Key Focus**: The model is optimized to **maximize fraud recall** — it is better to investigate a false alarm than to miss a real fraud.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/ArokiyaNithish/Fraud_Detection_Project.git
cd Fraud_Detection_Project
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** contents:

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
flask>=2.0.0
joblib>=1.1.0
```

### 4. Download Dataset

1. Visit [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in the `data/` folder

---

## 💻 Usage Guide

### 🔧 Training the Full Pipeline

Run the scripts in this order from the `scripts/` directory:

```bash
cd scripts

# Step 1: Preprocess data → creates creditcard_clean.csv and scaler.pkl
python preprocessing.py

# Step 2: Split into train/test → creates X_train.csv, X_test.csv, y_train.csv, y_test.csv
python split_data.py

# Step 3: Balance training data with SMOTE
python balance_data.py

# Step 4 (Optional): Train Logistic Regression baseline
python model_building.py

# Step 5: Train Random Forest (recommended) → creates random_forest_model.pkl
python random_forest_model.py

# Step 6: Evaluate model performance
python evaluate_model.py

# Step 7: Visualize feature importance
python feature_importance.py
```

**OR use the complete end-to-end training pipeline:**

```bash
python trained.py   # Runs SMOTE + RandomForest(200 trees) all-in-one
```

---

### 🖥️ Command-Line Prediction

```bash
python real_time_prediction.py
```

Runs a test transaction through the trained model and prints:
```
✅ Model and Scaler loaded successfully!
✅ LEGITIMATE TRANSACTION: This transaction seems safe.
Fraud Probability: 0.12%
```

---

## 🌐 Web Application

### Starting the Flask App

```bash
# From the project root:
cd app
python app.py
```

Then open your browser at:
```
http://127.0.0.1:5000/
```

### How to Use the Web Interface

1. **Enter Values** for V1 through V28 (PCA-transformed features)
2. **Enter Amount** (transaction amount in USD)
3. Click **"Check Fraud"**
4. View the result:

| Result | Display |
|--------|---------|
| Fraud Detected | 🚨 `Fraud Detected! (Probability: 87.43%)` |
| Legitimate | ✅ `Legitimate Transaction (Probability: 0.12%)` |
| Invalid Input | ❌ `Invalid input. Please enter valid numbers.` |

### Example Test Values (Known Fraud Transaction)

Use these V1–V28 values from the first fraudulent record in the dataset:

```
V1: -1.359807   V2: -0.072781   V3: 2.536346    V4: 1.378155
V5: -0.338321   V6: 0.462388    V7: 0.239599    V8: 0.098698
V9: 0.363787    V10: 0.090795   V11: -0.551600  V12: -0.617801
V13: -0.991390  V14: -0.311169  V15: 1.468177   V16: -0.470400
V17: 0.207971   V18: 0.025791   V19: 0.403993   V20: 0.251412
V21: -0.018307  V22: 0.277838   V23: -0.110474  V24: 0.066928
V25: 0.128539   V26: -0.189115  V27: 0.133558   V28: -0.021053
Amount: 149.62
```

---

## 🔍 Code Analysis

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SMOTE over Random Undersampling** | Preserves all legitimate transaction data while generating synthetic fraud samples |
| **Random Forest over SVM / Neural Nets** | Better performance with tabular data; natural feature importance; handles overfitting well |
| **`class_weight='balanced'` + SMOTE** | Double protection against imbalance — extra robustness |
| **Stratified split** | Ensures fraud examples appear in both train and test sets proportionally |
| **StandardScaler only on Amount** | V1–V28 are already PCA-scaled; only Amount needs normalization |
| **Scaler saved separately** | Ensures identical transformation at inference time (prevents data leakage) |
| **200 trees in final model** | More trees = more stability; diminishing returns beyond 200 for this dataset |

### Script Dependency Map

```
creditcard.csv
     ↓
preprocessing.py → creditcard_clean.csv + scaler.pkl
     ↓
split_data.py → X_train.csv, X_test.csv, y_train.csv, y_test.csv
     ↓
balance_data.py → X_train_balanced.csv, y_train_balanced.csv
     ↓
random_forest_model.py → random_forest_model.pkl
     ↓
evaluate_model.py → Metrics output
feature_importance.py → Plot
real_time_prediction.py → CLI predictions
app/app.py → Flask web server
```

---

## 🌍 Impact & Real-World Significance

### Economic Impact

- 💰 **$32 billion+** in annual global credit card fraud losses
- 🏦 **1 in 20** cardholders experiences fraud annually
- ⚡ This system can catch fraud **in milliseconds**, before transaction authorization

### Why This Matters

| Stakeholder | Benefit |
|-------------|---------|
| **Cardholders** | Protection of savings and financial identity |
| **Banks / Issuers** | Reduced chargebacks and financial losses |
| **Merchants** | Fewer fraudulent reversals impacting revenue |
| **Financial System** | Enhanced integrity and consumer trust |

### System Advantages Over Manual Review

| Manual Review | This ML System |
|---------------|---------------|
| Hours/days to investigate | **Milliseconds** to predict |
| Human error-prone | **Consistent, data-driven decisions** |
| Cannot scale | **Handles millions of transactions** |
| Expensive (labor) | **Low-cost once trained** |

---

## 🖼️ Screenshots

### Web Application Interface
<img width="789" height="829" alt="3" src="https://github.com/user-attachments/assets/b6a50d5b-5e28-410c-870e-3ae509c11220" />


### Prediction Results
<img width="776" height="55" alt="4" src="https://github.com/user-attachments/assets/37541036-f611-441d-90aa-d753f709e081" />

### Feature Importance Visualization
<img width="778" height="438" alt="result" src="https://github.com/user-attachments/assets/8236c98f-4bf5-4bc8-a871-c461e043bb5b" />

<img width="799" height="578" alt="1" src="https://github.com/user-attachments/assets/5757442c-5a9a-40cf-9e64-60ed143a09c3" />
---

# Working Prototype Video


https://github.com/user-attachments/assets/b9e64ce7-d734-4dcf-b6e3-d04dc4279c13


---

## 🚀 Future Enhancements

- [ ] **Deep Learning**: Implement LSTM/Autoencoder for anomaly detection on transaction sequences
- [ ] **XGBoost / LightGBM**: Compare gradient boosting models
- [ ] **Explainable AI (XAI)**: Integrate SHAP/LIME for per-prediction explanations
- [ ] **Real-time Dashboard**: Live transaction monitoring with Plotly Dash
- [ ] **REST API**: FastAPI or Flask-RESTful for bank system integration
- [ ] **Cloud Deployment**: Docker containerization + deploy on AWS/GCP/Heroku
- [ ] **Automated Retraining**: MLOps pipeline with scheduled model updates
- [ ] **Alert System**: Email/SMS notifications on fraud detection
- [ ] **Database Integration**: PostgreSQL/MongoDB for audit logs
- [ ] **Multi-model Ensemble**: Stack Random Forest + XGBoost + Neural Net

---

## 🤝 Open Source Contribution

We warmly welcome contributions from the open source community! Whether it's **bug fixes**, **new models**, **UI improvements**, or **documentation** — every contribution helps!

### How to Contribute

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Fraud_Detection_Project.git
cd Fraud_Detection_Project

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make changes and commit
git add .
git commit -m "feat: add XGBoost model comparison"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub → main branch
```

### Contribution Areas

| Area | Good First Issue? | Description |
|------|------------------|-------------|
| 🐛 **Bug Fixes** | ✅ Yes | Fix hardcoded file paths, edge cases |
| 📊 **New Models** | ✅ Yes | Add XGBoost, LightGBM, SVM comparison |
| 🌐 **UI Improvement** | ✅ Yes | Better Flask UI with CSS/Bootstrap |
| 📉 **Visualization** | ✅ Yes | Add ROC curve, PR curve plots |
| 🧪 **Unit Tests** | ✅ Yes | Add pytest test cases for scripts |
| 📖 **Documentation** | ✅ Yes | Improve docstrings, add tutorials |
| 🔧 **Config File** | ⚡ Medium | Replace hardcoded paths with config.yaml |
| ☁️ **Cloud Deploy** | ⚡ Medium | Dockerfile + deployment scripts |
| 🤖 **MLOps Pipeline** | 🔥 Advanced | GitHub Actions CI/CD for retraining |

### Coding Standards

- Follow **PEP 8** style guide
- Add **docstrings** to all functions/scripts
- Write **meaningful commit messages** (use [Conventional Commits](https://www.conventionalcommits.org/))
- Test your changes locally before submitting PR
- Reference any issue number in your PR description

### Reporting Issues

Please use [GitHub Issues](https://github.com/ArokiyaNithish/Fraud_Detection_Project/issues) to:
- 🐛 Report bugs
- 💡 Request features
- ❓ Ask questions

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this code with attribution.

```
MIT License

Copyright (c) 2025 Arokiya Nithish

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software...
```

See [LICENSE](LICENSE) for full details.

---

## 👨‍💻 Author & Acknowledgments

### Author

**Arokiya Nithish J**
- 🎓 Internship Company — Pratinik Infotech, Intern ID: 4138
- 📅 Year: 2025
- 💼 Domain: Machine Learning | Data Science | Financial Technology
- [Internship Completed Certificate](https://drive.google.com/file/d/1nI-2xUSE-KImk_2WDrnI2FfbeRPDl0uc/view?usp=sharing)
- [Internship Offer Letter](https://drive.google.com/file/d/10r7hiPup_ZBHaB4UwMsM8-cTb_Xr6bz3/view?usp=sharing)

**Contacts**
- GitHub: [@ArokiyaNithish](https://github.com/ArokiyaNithish)
- LinkedIn: [@Arokiya Nithish J](https://www.linkedin.com/in/arokiya-nithishj/)
- Email ID: @arokiyanithishj@gmail.com
- My Portfoilio: [@Arokiya Nithish](arokiyanithish.github.io/portfolio/)


### Acknowledgments

- 🏢 **Pratinik Infotech** — For providing this real-world problem statement and internship opportunity
- 📊 **Machine Learning Group (ULB)** — For the publicly available [Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) on Kaggle
- 🤖 **scikit-learn Community** — For the world-class, open-source ML library
- ⚖️ **imbalanced-learn Team** — For SMOTE and other class-balancing tools
- 🌐 **Flask Team** — For the lightweight, powerful web framework

---

## 📚 References

1. [Kaggle — Credit Card Fraud Detection Dataset (ULB)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. [scikit-learn — RandomForestClassifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
3. [imbalanced-learn — SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
4. [Flask — Official Documentation](https://flask.palletsprojects.com/)
5. [Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.](https://doi.org/10.1023/A:1010933404324)
6. [Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.](https://doi.org/10.1613/jair.953)
7. [Report: Fraud Detection in Credit Card Transactions* — Project Report, Pratinik Infotech Internship, 2025](https://drive.google.com/file/d/1vmGPNKVxIpgTs7T3EUIQesi_q2YGBiM7/view?usp=sharing)

---

<div align="center">

For support, email @arokiyanithishj@gmail.com or create an issue in the GitHub repository.

### 🌟 If this project helped you — please give it a ⭐ Star on GitHub!

**#MachineLearning #FraudDetection #Python #RandomForest #Flask #DataScience**

*Made with ❤️ and Python by Arokiya Nithish*

</div>
