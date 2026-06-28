"""
Project: AvgEGT Prediction (Regression Pipeline)
Script: regression_pipeline.py
Purpose: This is the core machine learning pipeline. It loads the raw engine telemetry data,
         cleans it, sanitizes it against 'Data Leakage', scales it, and tests 5 different 
         Machine Learning algorithms (Linear, Decision Tree, Random Forest, Extra Trees, XGBoost)
         to predict the exact continuous temperature (AvgEGT) of the engine exhaust.
Inputs:  `data/raw/AE_DATA_with_AvgEGT.csv`
Outputs: 1. `final_model.pkl` (The winning trained model saved for later inference)
         2. Multiple diagnostic images saved to the `plots/` directory (Model Comparison, 
            Actual vs Predicted, Feature Importance, Residuals, Correlation Heatmap).
"""
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for matplotlib to generate files silently without a UI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore') # Suppress non-critical warnings for a cleaner console output

# =====================================================================
# PHASE 1: CONFIGURATION & SETUP
# =====================================================================
# Define exactly where the raw data lives and where our plots will go
DATA_PATH = os.path.join("data", "raw", "AE_DATA_with_AvgEGT.csv")
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TARGET_COL is the exact mathematical value we are trying to predict
TARGET_COL = "AvgEGT"

# EXCLUDED_COLS defines variables that must be dropped. 
# Why? Because EXHAUST TEMP 1 through 6 perfectly average out to equal the Target (AvgEGT).
# If we let the AI see these columns, it will "cheat" by averaging them instead of learning
# the true physical relationships between pressure, load, and temperature.
EXCLUDED_COLS = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3",
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]

def main():
    print("===========================================")
    print(" AvgEGT Machine Learning Regression Project")
    print("===========================================\n")

    # =====================================================================
    # PHASE 2: DATA LOADING
    # =====================================================================
    # Verify the file actually exists before we try to open it
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at '{DATA_PATH}'.")
        return

    df = pd.read_csv(DATA_PATH)
    original_shape = df.shape
    print(f"Loaded dataset: {original_shape[0]} rows, {original_shape[1]} columns")

    # =====================================================================
    # PHASE 3: DATA CLEANING & PREPROCESSING
    # =====================================================================
    # Remove exact duplicate rows to prevent the model from memorizing duplicate data
    df = df.drop_duplicates()
    
    # Drop any row that is missing data. AI algorithms cannot process blank (NaN) cells.
    df = df.dropna()

    # Filter out extreme anomalies. In this mechanical context, temperatures above 1000°C 
    # are physically impossible or represent catastrophic failure outside normal prediction bounds.
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL] <= 1000]

    # Dynamically find which of our excluded columns actually exist in the current dataset and drop them
    drop_cols = [c for c in EXCLUDED_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Hard-stop if our target variable somehow got deleted
    if TARGET_COL not in df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in dataset.")
        return

    # Split the dataset: 'X' is the mechanical features (inputs), 'y' is the temperature (output)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    final_features = list(X.columns)

    print(f"\n[Data Summary]")
    print(f"- Rows used for training: {len(df)}")
    print(f"- Excluded features: {drop_cols}")
    print(f"- Final features ({len(final_features)}): {final_features}")

    # =====================================================================
    # PHASE 4: TRAIN-TEST SPLIT & SCALING
    # =====================================================================
    # We hide 10% of the data (the 'test' set) in a vault. The AI will only train on 90%.
    # Later, we will test the AI on the 10% it has never seen to prove it didn't just memorize the answers.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Scaling: Machine learning models get confused if one column has massive numbers (like KW at 3200)
    # and another has tiny numbers (like Pressure at 3.5). StandardScaler crushes all numbers 
    # down to a mathematically equal playing field (z-scores).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Transform the test set using the train set's rules!
    
    # Re-wrap the raw arrays back into Pandas DataFrames so we don't lose our column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=final_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=final_features, index=X_test.index)

    # =====================================================================
    # PHASE 5: MODEL TRAINING & COMPARISON
    # =====================================================================
    print("\n[Training & Comparing Models]")
    
    # We initialize a dictionary of multiple algorithms. It's best practice to run a "Shootout"
    # because different math architectures perform differently depending on the data shape.
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=42, n_jobs=-1),
        # XGBoost is the heavy hitter. It builds 500 small trees sequentially, each correcting the previous tree's mistakes.
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    }

    results = []
    best_model_name = None
    best_r2 = -float('inf')
    best_model_instance = None
    best_y_pred = None

    print(f"{'Model':<20} | {'R²':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 57)

    for name, model in models.items():
        # Train the algorithm
        model.fit(X_train_scaled, y_train)
        
        # Ask it to predict the temperatures of the 10% test vault data
        y_pred = model.predict(X_test_scaled)
        
        # Calculate how right (or wrong) it was
        r2 = r2_score(y_test, y_pred) # Percentage of variance explained (Higher is better)
        mae = mean_absolute_error(y_test, y_pred) # Average degrees it was off by (Lower is better)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Punishes massive outlier mistakes (Lower is better)
        
        results.append({'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
        print(f"{name:<20} | {r2:<10.4f} | {mae:<10.4f} | {rmse:<10.4f}")

        # Keep track of whoever scored the highest R²
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_instance = model
            best_y_pred = y_pred

    results_df = pd.DataFrame(results)

    # =====================================================================
    # PHASE 6: FINAL MODEL SELECTION & SAVING
    # =====================================================================
    print(f"\n[Final Model Selection]")
    
    # 90% accuracy is the business threshold for this project
    if best_r2 >= 0.90:
        print(f"Success! Model crossed the 90% threshold with R²: {best_r2:.4f}")
        
    print(f"Selected Model: {best_model_name}")
    print(f"Test Set R²:    {r2_score(y_test, best_y_pred):.4f}")
    
    # Serialize (save) the winning brain into a binary .pkl file so we can load it instantly later 
    # in our predict_new_engine_data_REGRESSION.py script without re-training!
    model_path = "final_model.pkl"
    joblib.dump(best_model_instance, model_path)
    print(f"Model saved to: {model_path}")

    # =====================================================================
    # PHASE 7: VISUALIZATION & PLOTTING
    # =====================================================================
    print("\n[Generating Plots]")
    
    # --- 7.1 Model Comparison Plot ---
    plt.figure(figsize=(10, 6))
    sorted_df = results_df.sort_values('R2', ascending=False)
    
    # Visually pop the winner in green, gray out the losers
    colors = ['limegreen' if model == best_model_name else 'lightgray' for model in sorted_df['Model']]
    
    ax = sns.barplot(data=sorted_df, x='Model', y='R2', palette=colors)
    plt.title('Model Comparison - R² Score (Highlighting the Best)')
    plt.ylabel('R² Score')
    plt.ylim(0, 1.1)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.4f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontweight='bold' if i == 0 else 'normal')
        
        if i == 0:
            ax.annotate('⭐ BEST MODEL', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 25), 
                        textcoords='offset points', color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_model_comparison.png'))
    plt.close()

    # --- 7.2 Actual vs Predicted Plot ---
    # Proves the model is accurate. A perfect model would form a solid straight red line.
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, best_y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted ({best_model_name})')
    plt.xlabel('Actual AvgEGT')
    plt.ylabel('Predicted AvgEGT')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_actual_vs_predicted.png'))
    plt.close()

    # --- 7.3 Feature Importance Plot ---
    # Asks the algorithm: "Which column did you rely on most heavily to make your decisions?"
    if hasattr(best_model_instance, 'feature_importances_'):
        importances = best_model_instance.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [final_features[i] for i in indices]

        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette='mako', legend=False)
        plt.title('Feature Importances')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '03_feature_importance.png'))
        plt.close()

    # --- 7.4 Residual Plot ---
    # Plots the "mistakes". If the dots are randomly scattered around the zero line, our math is sound.
    # If they form a distinct shape (like a U), we are missing a non-linear variable.
    residuals = y_test - best_y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(best_y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted AvgEGT')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_residuals.png'))
    plt.close()

    # --- 7.5 Correlation Heatmap ---
    # Shows the raw mathematical relationship between all features. Dark red = highly correlated.
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_correlation_heatmap.png'))
    plt.close()

    print(f"Successfully generated 5 diagnostic plots in the '{OUTPUT_DIR}' directory.")
    print("\n--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()
