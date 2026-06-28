"""
Project: AvgEGT Prediction (Classification Pipeline)
Script: classification_pipeline.py
Purpose: This is the critical safety pipeline. Instead of predicting an exact temperature, it reframes
         the math into a Binary Classification problem. It isolates the top 10% hottest engine states
         and trains models to flag them as a '1' (Critical Risk). It heavily utilizes class_weight
         balancing to force the AI to respect the imbalanced data (90% normal, 10% failure).
Outputs: 1. `classification_model.pkl` (The winning trained classifier)
         2. `classification_plots/` containing Accuracy Comparisons and Confusion Matrices.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Silent plotting backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore') # Keep console clean

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =====================================================================
# PHASE 1: CONFIGURATION & SETUP
# =====================================================================
# Step back out of 'classification_project' folder using ".." to grab the raw dataset
DATA_PATH = os.path.join("..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
OUTPUT_DIR = "classification_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "AvgEGT"

# Same data leakage rules apply here. We must hide the answers from the AI.
excluded_cols = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3",
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]

def main():
    print("===========================================")
    print(" AvgEGT BINARY CLASSIFICATION PROJECT")
    print(" Goal: Predict 'Critical Risk' (Top 10% Temps)")
    print("===========================================\n")

    # =====================================================================
    # PHASE 2: DATA LOADING & SANITIZATION
    # =====================================================================
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at '{DATA_PATH}'. Please run this script from inside the classification_project folder.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Drop duplicates and blank cells to ensure pure data
    df = df.drop_duplicates().dropna()
    
    # Remove extreme anomalies (physical impossibilities > 1000C)
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL] <= 1000]

    # Drop the excluded columns to prevent data leakage
    drop_cols = [c for c in excluded_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # =====================================================================
    # PHASE 3: TRANSFORM TO BINARY CLASSIFICATION 
    # =====================================================================
    # We define "Critical" mathematically as the 90th percentile (the top 10% hottest engine states)
    threshold = np.percentile(df[TARGET_COL], 90)
    print(f"Defining 'Critical Risk' as AvgEGT >= {threshold:.2f}°C")

    # Create a new binary target column: 1 if Critical (above threshold), 0 if Normal
    df['Risk_Class'] = (df[TARGET_COL] >= threshold).astype(int)
    
    # We drop the actual temperature value entirely. The AI is no longer predicting degrees, it is predicting 1s and 0s.
    X = df.drop(columns=[TARGET_COL, 'Risk_Class'])
    y = df['Risk_Class']

    # Show the imbalance. The data will be roughly 90% Normal (0) and 10% Critical (1).
    print(f"Class Distribution:\n{y.value_counts(normalize=True)*100}\n")

    # =====================================================================
    # PHASE 4: TRAIN-TEST SPLIT & SCALING
    # =====================================================================
    # stratify=y is CRITICAL here. Because only 10% of our data is "Critical", if we split randomly,
    # the 10% test vault might accidentally get zero critical failures! Stratify guarantees the test vault
    # has the exact same 90/10 ratio as the training set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Standardize the features so large sensor readings don't overpower small ones
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # =====================================================================
    # PHASE 5: MODEL TRAINING & COMPARISON
    # =====================================================================
    print("[Training & Comparing Classification Models]")
    
    # Notice that we pass `class_weight='balanced'` to almost every model.
    # Why? Because if 90% of the data is Normal, a lazy AI could just guess "Normal" every single time
    # and achieve a "90% Accuracy" without doing any actual math. `class_weight` mathematically forces 
    # the AI to multiply its internal penalty by 9x anytime it fails to catch a Critical (1) failure.
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight='balanced', n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1, class_weight='balanced')
    }

    results = []
    best_model_name = None
    best_acc = -1
    best_model_instance = None
    best_y_pred = None

    print(f"{'Model':<20} | {'Accuracy':<10}")
    print("-" * 33)

    for name, model in models.items():
        # Train
        model.fit(X_train_s, y_train)
        # Predict the 10% test vault
        y_pred = model.predict(X_test_s)
        
        # Calculate raw accuracy (Percentage of time it guessed right)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({'Model': name, 'Accuracy': acc})
        print(f"{name:<20} | {acc:<10.4f}")

        # Keep track of the winner
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_instance = model
            best_y_pred = y_pred

    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

    # =====================================================================
    # PHASE 6: FINAL SELECTION & ANALYSIS
    # =====================================================================
    print(f"\n[Final Classification Model]")
    print(f"Selected Model: {best_model_name}")
    print(f"Test Set Accuracy: {best_acc*100:.2f}%")
    if best_acc >= 0.95:
        print(">>> 95% ACCURACY GOAL ACHIEVED! <<<")
        
    print("\n[Classification Report]")
    # The classification report prints Precision (Zero False Alarms?) and Recall (Did we catch all failures?)
    print(classification_report(y_test, best_y_pred, target_names=["Normal", "Critical"]))

    # Serialize (save) the winning brain
    joblib.dump(best_model_instance, "classification_model.pkl")

    # =====================================================================
    # PHASE 7: CLASSIFICATION PLOTS
    # =====================================================================
    
    # --- 7.1 Accuracy Comparison Plot ---
    plt.figure(figsize=(10, 6))
    colors = ['limegreen' if m == best_model_name else 'lightgray' for m in results_df['Model']]
    ax = sns.barplot(data=results_df, x='Model', y='Accuracy', palette=colors)
    plt.title('Classification Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height()*100:.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_accuracy_comparison.png'))
    plt.close()

    # --- 7.2 Confusion Matrix ---
    # The Confusion Matrix is the ultimate truth-teller for classification. 
    # It shows True Positives (caught failure), True Negatives (safely normal),
    # False Positives (False Alarm!), and False Negatives (Catastrophic Miss!).
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Normal", "Critical"], yticklabels=["Normal", "Critical"])
    plt.title(f'Actual Confusion Matrix ({best_model_name})')
    plt.xlabel('Predicted Risk Class')
    plt.ylabel('Actual Risk Class')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_confusion_matrix.png'))
    plt.close()

    print(f"\nPlots saved in '{OUTPUT_DIR}'")
    print("--- Classification Pipeline Complete ---")

if __name__ == "__main__":
    main()
