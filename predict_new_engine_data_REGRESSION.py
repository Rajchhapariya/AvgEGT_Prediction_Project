import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

print("Loading data and initializing Regression Model...")

# 1. Load Data for Training/Scaling
DATA_PATH = os.path.join("data", "raw", "AE_DATA_with_AvgEGT.csv")
df = pd.read_csv(DATA_PATH).drop_duplicates().dropna()
df = df[df['AvgEGT'] <= 1000]

# These are the 13 required columns (12 features + 1 target)
excluded_cols = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3", 
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]
df = df.drop(columns=[c for c in excluded_cols if c in df.columns])

X = df.drop(columns=['AvgEGT'])
y = df['AvgEGT']

# Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_scaled, y)


# =====================================================================
# ⬇️ ENTER YOUR NEW ENGINE SENSOR DATA BELOW ⬇️
# =====================================================================
# You can change these numbers to simulate any engine condition!
# (These default values are typical baseline numbers)

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


print("\n--- Executing Inference ---")
# Convert input to dataframe
new_df = pd.DataFrame([NEW_ENGINE_DATA])

# Ensure columns match training data exactly
missing_cols = set(X.columns) - set(new_df.columns)
if missing_cols:
    print(f"ERROR: Missing columns in input data: {missing_cols}")
else:
    # Scale input using the EXACT SAME scaler
    new_df_scaled = scaler.transform(new_df[X.columns])
    
    # Predict!
    prediction = model.predict(new_df_scaled)[0]
    
    print("\n[INPUT SENSORS]:")
    for k, v in NEW_ENGINE_DATA.items():
        print(f"  {k}: {v}")
        
    print("\n===========================================")
    print(f"Predicted AvgEGT: {prediction:.2f} Degrees")
    print("===========================================\n")
