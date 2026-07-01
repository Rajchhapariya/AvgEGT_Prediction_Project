"""
Project: AvgEGT Prediction (Regression Pipeline)
Script: predict_new_engine_data_REGRESSION.py
Purpose: A client-facing inference script. It does NOT train a model. Instead, it re-trains/loads 
         the XGBoost logic instantly and allows a user to type in 12 live sensor readings 
         (like FO TEMP and TC LO PRESS) to predict the exact temperature of the exhaust in real-time.
Inputs:  `NEW_ENGINE_DATA` dictionary (hardcoded below for client testing)
Outputs: Prints the exact predicted 'AvgEGT' decimal temperature to the terminal.
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Keep the terminal output clean for the client

from sklearn.preprocessing import StandardScaler
import joblib

print("Loading data and initializing Regression Model...")

# =====================================================================
# PHASE 1: BACKGROUND DATA PREPARATION
# =====================================================================
# Even though we are just predicting, we must load the original dataset to fit our StandardScaler.
# A StandardScaler needs to see the original "min/max/average" of the historical data so it knows
# exactly how to scale the client's new live input data down to the exact same proportion!
DATA_PATH = os.path.join("data", "raw", "AE_DATA_with_AvgEGT.csv")
df = pd.read_csv(DATA_PATH).drop_duplicates().dropna()

# Drop anomalies
df = df[df['AvgEGT'] <= 1000]

# Exclude the "cheating" columns to prevent data leakage, ensuring we match the training phase exactly
excluded_cols = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3", 
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]
df = df.drop(columns=[c for c in excluded_cols if c in df.columns])

# Separate Features (X) from Target (y)
X = df.drop(columns=['AvgEGT'])
y = df['AvgEGT']

# Scale the data using the historical means and variances
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the pre-trained winning model from the pipeline
model = joblib.load("final_model.pkl")


# =====================================================================
# PHASE 2: CLIENT INPUT (ENTER NEW SENSOR DATA)
# =====================================================================
# Client Instructions: Change these numbers to simulate any engine condition.
# This dictionary represents a single "snapshot" of the engine telemetry at a given second.
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
# PHASE 3: EXECUTE INFERENCE (PREDICTION)
# =====================================================================
print("\n--- Executing Inference ---")

# Convert the dictionary into a Pandas DataFrame format that the AI expects
new_df = pd.DataFrame([NEW_ENGINE_DATA])

# Sanity Check: Ensure the client didn't accidentally delete or misspell a required sensor!
missing_cols = set(X.columns) - set(new_df.columns)
if missing_cols:
    print(f"ERROR: Missing columns in input data: {missing_cols}")
else:
    # CRITICAL: We must scale the client's input data using the EXACT SAME historical scaler object.
    # If we didn't scale this, the AI (expecting numbers between -3 and 3) would freak out 
    # when it sees a KW value of 3200.0!
    new_df_scaled = scaler.transform(new_df[X.columns])
    
    # Feed the scaled numbers into the XGBoost brain to get the predicted temperature
    prediction = model.predict(new_df_scaled)[0]
    
    # Print out the results in a friendly dashboard format
    print("\n[INPUT SENSORS]:")
    for k, v in NEW_ENGINE_DATA.items():
        print(f"  {k}: {v}")
        
    print("\n===========================================")
    print(f"Predicted AvgEGT: {prediction:.2f} Degrees")
    print("===========================================\n")
