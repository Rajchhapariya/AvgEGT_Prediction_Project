import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("Loading data and initializing Classification Model...")

# 1. Load Data for Training/Scaling/Thresholding
DATA_PATH = os.path.join("..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
# Fallback in case they run it from root instead of classification_project folder
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("data", "raw", "AE_DATA_with_AvgEGT.csv")

df = pd.read_csv(DATA_PATH).drop_duplicates().dropna()
df = df[df['AvgEGT'] <= 1000]

excluded_cols = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3", 
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]
df = df.drop(columns=[c for c in excluded_cols if c in df.columns])

X = df.drop(columns=['AvgEGT'])
y_reg = df['AvgEGT']

# 2. Calculate the 90th Percentile Risk Threshold dynamically
threshold = np.percentile(y_reg, 90)
y_cls = (y_reg >= threshold).astype(int)

# Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_scaled, y_cls)


# =====================================================================
# ⬇️ ENTER YOUR NEW ENGINE SENSOR DATA BELOW ⬇️
# =====================================================================
# You can change these numbers to simulate any engine condition!
# Example: To trigger a Failure Warning, push TC LO PRESS and FO TEMP 
# way beyond their normal limits!

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


print("\n--- Executing Risk Classification ---")
new_df = pd.DataFrame([NEW_ENGINE_DATA])

missing_cols = set(X.columns) - set(new_df.columns)
if missing_cols:
    print(f"ERROR: Missing columns in input data: {missing_cols}")
else:
    # Scale
    new_df_scaled = scaler.transform(new_df[X.columns])
    
    # Predict Label and Probability
    prediction = model.predict(new_df_scaled)[0]
    probability = model.predict_proba(new_df_scaled)[0][1] * 100
    
    print("\n[INPUT SENSORS]:")
    for k, v in NEW_ENGINE_DATA.items():
        print(f"  {k}: {v}")
        
    print("\n===========================================")
    if prediction == 1:
        print(f"Risk Status: CRITICAL ALARM TRIGGERED")
    else:
        print(f"Risk Status: NORMAL")
    
    print(f"Confidence of Engine Failure: {probability:.2f}%")
    print("===========================================\n")
