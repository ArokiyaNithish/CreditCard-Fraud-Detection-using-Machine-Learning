# step8_feature_importance.py

import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Step 1: Load trained Random Forest model
rf_model = joblib.load("random_forest_model.pkl")
print("Random Forest model loaded successfully!")

# Step 2: Load one of the datasets to get feature names
X_train = pd.read_csv("X_train_balanced.csv")
feature_names = X_train.columns

# Step 3: Get feature importance from the model
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Step 4: Display top 10 features
print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Step 5: Plot feature importance
plt.figure(figsize=(12,6))
plt.barh(importance_df['Feature'].head(10), importance_df['Importance'].head(10), color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features for Fraud Detection")
plt.gca().invert_yaxis()
plt.show()
