import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------
# Step back out of 'classification_project' folder to grab the data
DATA_PATH = os.path.join("..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
OUTPUT_DIR = "classification_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "AvgEGT"
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

    # ---------------------------------------------------------
    # 2. Data Loading & Sanitization
    # ---------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at '{DATA_PATH}'. Please run this script from inside the classification_project folder.")
        return

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().dropna()
    
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL] <= 1000]

    drop_cols = [c for c in excluded_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ---------------------------------------------------------
    # 3. Transform to Binary Classification (95% Accuracy Target)
    # ---------------------------------------------------------
    # We define "Critical" as the top 10% of temperatures.
    threshold = np.percentile(df[TARGET_COL], 90)
    print(f"Defining 'Critical Risk' as AvgEGT >= {threshold:.2f}°C")

    # Create new target: 1 if Critical, 0 if Normal
    df['Risk_Class'] = (df[TARGET_COL] >= threshold).astype(int)
    
    X = df.drop(columns=[TARGET_COL, 'Risk_Class'])
    y = df['Risk_Class']

    print(f"Class Distribution:\n{y.value_counts(normalize=True)*100}\n")

    # ---------------------------------------------------------
    # 4. Train-Test Split & Scaling
    # ---------------------------------------------------------
    # Using stratify=y to ensure the test set has a balanced proportion of critical cases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 5. Model Training & Comparison
    # ---------------------------------------------------------
    print("[Training & Comparing Classification Models]")
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
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({'Model': name, 'Accuracy': acc})
        print(f"{name:<20} | {acc:<10.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_instance = model
            best_y_pred = y_pred

    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

    # ---------------------------------------------------------
    # 6. Final Selection & Analysis
    # ---------------------------------------------------------
    print(f"\n[Final Classification Model]")
    print(f"Selected Model: {best_model_name}")
    print(f"Test Set Accuracy: {best_acc*100:.2f}%")
    if best_acc >= 0.95:
        print(">>> 95% ACCURACY GOAL ACHIEVED! <<<")
        
    print("\n[Classification Report]")
    print(classification_report(y_test, best_y_pred, target_names=["Normal", "Critical"]))

    # Save Model
    joblib.dump(best_model_instance, "classification_model.pkl")

    # ---------------------------------------------------------
    # 7. Classification Plots
    # ---------------------------------------------------------
    # 7.1 Accuracy Comparison
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

    # 7.2 Real Confusion Matrix
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
