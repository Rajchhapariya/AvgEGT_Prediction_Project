import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
# Assuming execution from within classification_project/models/
DATA_PATH = os.path.join("..", "..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
df = pd.read_csv(DATA_PATH).drop_duplicates().dropna()

if 'AvgEGT' in df.columns:
    df = df[df['AvgEGT'] <= 1000]

# 2. Exclude Features
excluded_cols = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3", 
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]
drop_cols = [c for c in excluded_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# 3. Transform to Binary Classification (Top 10% Critical Risk)
threshold = np.percentile(df['AvgEGT'], 90)
df['Risk_Class'] = (df['AvgEGT'] >= threshold).astype(int)

X = df.drop(columns=['AvgEGT', 'Risk_Class'])
y = df['Risk_Class']

# 4. Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

# 5. Initialize Model
model = LogisticRegression(random_state=42, class_weight='balanced')

# 6. Train and Evaluate

model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print("--- Model Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Critical"]))
