"""
Project: AvgEGT Prediction (Classification Pipeline)
Script: predict_new_engine_data_CLASSIFICATION.py
Purpose: A client-facing inference script. It acts as a live alarm system.
         The client inputs 12 live sensor readings, and the XGBoost Classification AI calculates 
         both a binary decision (NORMAL vs CRITICAL ALARM) and the exact % probability of engine failure.
Inputs:  `NEW_ENGINE_DATA` dictionary (hardcoded below for testing).
Outputs: Prints 'CRITICAL ALARM TRIGGERED' or 'NORMAL', along with the Confidence %.
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Keep the output clean for the client

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("Loading data and initializing Classification Model...")

# =====================================================================
# PHASE 1: BACKGROUND DATA PREPARATION & THRESHOLDING
# =====================================================================
# We load the data not to train, but to fit our StandardScaler so it knows the historical 
# bounds of the features, and to calculate the 90th percentile threshold dynamically.

# Step back out of 'classification_project' folder using ".." to grab the raw dataset
DATA_PATH = os.path.join("..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
# Fallback in case the user runs the script from the root directory instead of the sub-folder
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("data", "raw", "AE_DATA_with_AvgEGT.csv")

# Clean the historical rows
df = pd.read_csv(DATA_PATH).drop_duplicates().dropna()
df = df[df['AvgEGT'] <= 1000]

# Exclude cheating columns to prevent Data Leakage
excluded_cols = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3", 
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]
df = df.drop(columns=[c for c in excluded_cols if c in df.columns])

# Separate Features (X) from Target (y_reg)
X = df.drop(columns=['AvgEGT'])
y_reg = df['AvgEGT']

# Calculate the 90th Percentile Risk Threshold mathematically (Top 10% hottest engine states)
threshold = np.percentile(y_reg, 90)
# Convert the continuous temperatures into 1s (Critical) and 0s (Normal)
y_cls = (y_reg >= threshold).astype(int)

# Scale the inputs based on historical variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Re-initialize the winning XGBoost classifier
model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_scaled, y_cls)


# =====================================================================
# PHASE 2: CLIENT INPUT (ENTER NEW SENSOR DATA)
# =====================================================================
# Client Instructions: Change these numbers to simulate any engine condition.
# Example: To purposely trigger a "CRITICAL ALARM", push TC LO PRESS down and FO TEMP up 
# way beyond their normal mechanical limits!

NEW_ENGINE_DATA = {
    'KW': 3200.0,
    'LO IN TEMP': 48.0,
    'SCAV TEMP': 45.2,
    'FO TEMP': 110.0,
    'TC IN TEMP': 450.0,
    'TC OUT TEMP': 350.0,
    'LO PRESS': 45.5,
    'FO PRESS': 600.0,
    'CW PRESS': 3.5,
    'SCAV AIR PRESS': 1.8,
    'TC LO PRESS': 140.5,
    'Rack Index': 25.0
}
# =====================================================================


# =====================================================================
# PHASE 3: EXECUTE RISK CLASSIFICATION (INFERENCE)
# =====================================================================
print("\n--- Executing Risk Classification ---")
new_df = pd.DataFrame([NEW_ENGINE_DATA])

# Sanity Check
missing_cols = set(X.columns) - set(new_df.columns)
if missing_cols:
    print(f"ERROR: Missing columns in input data: {missing_cols}")
else:
    # CRITICAL: Scale the new data using the same rules as the historical data
    new_df_scaled = scaler.transform(new_df[X.columns])
    
    # Predict Label (0 or 1)
    prediction = model.predict(new_df_scaled)[0]
    
    # predict_proba returns an array [Probability_of_0, Probability_of_1]. 
    # We grab index [1] to see exactly how confident the AI is that the engine will fail.
    probability = model.predict_proba(new_df_scaled)[0][1] * 100
    
    print("\n[INPUT SENSORS]:")
    for k, v in NEW_ENGINE_DATA.items():
        print(f"  {k}: {v}")
        
    print("\n===========================================")
    # Translate the binary 1/0 into an English alarm status
    if prediction == 1:
        print(f"Risk Status: CRITICAL ALARM TRIGGERED")
    else:
        print(f"Risk Status: NORMAL")
    
    print(f"Confidence of Engine Failure: {probability:.2f}%")
    print("===========================================\n")
