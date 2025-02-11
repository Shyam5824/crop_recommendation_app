# train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# For saving the model
import joblib

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Initial Exploration
crop = pd.read_csv("dataSet/Crop_recommendation.csv")

# Check for missing values
if crop.isnull().sum().any():
    print("Dataset contains missing values. Handling missing values...")
    crop = crop.dropna()  # or handle appropriately

# Check for duplicated values
crop = crop.drop_duplicates()

# 2. Data Preprocessing
# Encoding the target variable
le = LabelEncoder()
crop['crop_no'] = le.fit_transform(crop['label'])
print("Label Encoding Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Drop the original label
crop.drop('label', axis=1, inplace=True)

# Define features and target
X = crop.drop('crop_no', axis=1)
y = crop['crop_no']

# Train-Test Split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training and Evaluation

# Initialize classifiers
rfc = RandomForestClassifier(random_state=42)

# Hyperparameter tuning for Random Forest
param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rfc = GridSearchCV(rfc, param_grid_rfc, cv=5, scoring='accuracy', n_jobs=-1)
grid_rfc.fit(X_train_scaled, y_train)
best_rfc = grid_rfc.best_estimator_
print(f"Best Random Forest Params: {grid_rfc.best_params_}")

# Evaluate Random Forest
y_pred_rfc = best_rfc.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rfc))

# 4. Save Model and Preprocessing Objects
import os

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save the trained model
joblib.dump(best_rfc, os.path.join(model_dir, "best_rfc.pkl"))
# Save the scaler
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
# Save the label encoder
joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

print("Model and preprocessing objects have been saved successfully.")