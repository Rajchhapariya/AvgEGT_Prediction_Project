import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('../data/raw/AE_DATA_with_AvgEGT.csv')
df = df.drop_duplicates().dropna()
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

# 3. Split & Scale
X = df.drop(columns=['AvgEGT'])
y = df['AvgEGT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 4. Initialize Keras Model
tf.random.set_seed(42)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_s.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Train and Evaluate

model.fit(X_train_s, y_train, epochs=200, batch_size=32, verbose=0)
y_pred = model.predict(X_test_s, verbose=0).flatten()

print("--- Keras Deep NN Results ---")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE:      {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
