"""
Project: AvgEGT Prediction (Utility)
Script: generate_exhaustive_plots.py
Purpose: A massive batch-rendering engine. Instead of outputting to a Jupyter Notebook,
         this script silently computes and saves over 200+ distinct statistical plots
         (Distributions, Boxplots, SHAP charts) directly to folders on the local hard drive.
Outputs: Creates multiple subdirectories inside `generated_exhaustive_plots/`.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend (generates images silently in the background)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
import shap
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# PHASE 1: CONFIGURATION & SETUP
# =====================================================================
DATA_PATH = os.path.join("data", "raw", "AE_DATA_with_AvgEGT.csv")
OUTPUT_BASE = "generated_exhaustive_plots"

# Subdirectories to structurally organize the 220+ images
DIRS = {
    "full_eda": os.path.join(OUTPUT_BASE, "full_dataset_eda"),
    "sanitized_eda": os.path.join(OUTPUT_BASE, "sanitized_dataset_eda"),
    "winning_model": os.path.join(OUTPUT_BASE, "winning_model_regression"),
    "executive": os.path.join(OUTPUT_BASE, "advanced_executive_plots"),
    "interpretability": os.path.join(OUTPUT_BASE, "model_interpretability")
}

# Ensure all folders exist before saving to them
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

TARGET_COL = "AvgEGT"
EXCLUDED_COLS = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3",
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]

def main():
    """
    Main orchestrator for generating all ~220 plots across different stages
    of the machine learning pipeline.
    """
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}.")
        return
        
    df_raw = pd.read_csv(DATA_PATH)
    
    # =====================================================================
    # PHASE 2: FULL DATASET EDA BATCH
    # =====================================================================
    # Generate charts for the absolutely raw data (including blanks and duplicates)
    print("Generating Full Dataset EDA plots (this may take a moment)...")
    generate_feature_loops(df_raw, DIRS["full_eda"], prefix="full")
    
    # =====================================================================
    # PHASE 3: DATA CLEANING & SANITIZATION
    # =====================================================================
    df_clean = df_raw.drop_duplicates().dropna()
    if TARGET_COL in df_clean.columns:
        df_clean = df_clean[df_clean[TARGET_COL] <= 1000]
        
    drop_cols = [c for c in EXCLUDED_COLS if c in df_clean.columns]
    df_clean = df_clean.drop(columns=drop_cols)
    
    # Generate charts for the perfectly clean data
    print("Generating Sanitized Dataset EDA plots...")
    generate_feature_loops(df_clean, DIRS["sanitized_eda"], prefix="sanitized")
    
    # =====================================================================
    # PHASE 4: DATA PREPARATION & MODEL TRAINING
    # =====================================================================
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    
    # Keep as DataFrames to retain feature names (Critical for SHAP labels!)
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    best_model, best_name, best_y_pred, results_df = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # =====================================================================
    # PHASE 5: EXECUTIVE & DIAGNOSTIC BATCH PLOTTING
    # =====================================================================
    print("Generating Winning Model plots...")
    generate_winning_model_plots(y_test, best_y_pred, best_name)
    
    print("Generating Advanced Executive plots...")
    generate_executive_plots(best_model, results_df, X.columns, df_clean)
    
    # =====================================================================
    # PHASE 6: SHAP (GAME THEORY) INTERPRETABILITY
    # =====================================================================
    print("Generating Model Interpretability (SHAP) plots...")
    generate_shap_plots(best_model, X_train_scaled)
    
    print("\n=======================================================")
    print(f"All exhaustive plots generated successfully in: {OUTPUT_BASE}")
    print("=======================================================")


def generate_feature_loops(df, out_dir, prefix=""):
    """
    Loops through every feature in the dataset and creates three plots for each:
    1. Distribution (Histogram)
    2. Boxplot (Outlier detection)
    3. Scatter plot vs Target
    """
    features = [c for c in df.columns if c != TARGET_COL]
    
    for i, feature in enumerate(features):
        # 1. Feature Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, color='teal')
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{prefix}_{feature}_dist.png"))
        plt.close()
        
        # 2. Boxplot
        plt.figure(figsize=(8, 2))
        sns.boxplot(x=df[feature], color='orange')
        plt.title(f'Boxplot of {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{prefix}_{feature}_box.png"))
        plt.close()
        
        # 3. Scatter vs Target
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=feature, y=TARGET_COL, alpha=0.6, color='navy')
        plt.title(f'{feature} vs {TARGET_COL}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{prefix}_{feature}_scatter.png"))
        plt.close()

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains models and returns the best model and performance dataframe.
    """
    models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = []
    best_r2 = -float('inf')
    best_model = None
    best_name = ""
    best_y_pred = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({'Model': name, 'R2': r2, 'RMSE': rmse})
        
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = model
            best_y_pred = y_pred
            
    return best_model, best_name, best_y_pred, pd.DataFrame(results)

def generate_winning_model_plots(y_test, y_pred, model_name):
    """
    Generates specific plots analyzing the performance of the winning model.
    """
    out_dir = DIRS["winning_model"]
    
    # Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted ({model_name})')
    plt.xlabel('Actual AvgEGT')
    plt.ylabel('Predicted AvgEGT')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_actual_vs_predicted.png'))
    plt.close()

    # Residuals vs Predicted
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted AvgEGT')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_residuals_plot.png'))
    plt.close()
    
    # Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color='salmon')
    plt.title('Error Distribution (Residuals)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_error_distribution.png'))
    plt.close()

def generate_executive_plots(best_model, results_df, feature_names, df_clean):
    """
    Generates high-level summary charts for executive review.
    """
    out_dir = DIRS["executive"]
    
    # Model Comparison Scoreboard
    plt.figure(figsize=(8, 6))
    sns.barplot(data=results_df, x='Model', y='R2', palette='viridis')
    plt.title('Model R² Performance Scoreboard')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_model_r2_comparison.png'))
    plt.close()
    
    # Global Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='mako')
        plt.title('Executive Summary: Top Drivers of AvgEGT')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '02_feature_importance.png'))
        plt.close()

    # Advanced Correlation Heatmap
    plt.figure(figsize=(14, 12))
    corr = df_clean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='vlag', annot=False)
    plt.title('Executive Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_executive_correlation.png'))
    plt.close()

def generate_shap_plots(model, X_train_scaled):
    """
    Uses the SHAP (SHapley Additive exPlanations) library to generate
    model interpretability plots.
    """
    out_dir = DIRS["interpretability"]
    
    try:
        # SHAP requires a TreeExplainer for tree-based models like Random Forest/Extra Trees
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values for a subset to save time (or full if dataset is small)
        shap_values = explainer.shap_values(X_train_scaled)
        
        # 1. SHAP Summary Plot (Bar) - Global Importance
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_train_scaled, plot_type="bar", show=False)
        plt.title('SHAP Global Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '01_shap_summary_bar.png'))
        plt.close()
        
        # 2. SHAP Summary Plot (Dot) - Directional Impact
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_train_scaled, show=False)
        plt.title('SHAP Feature Impact (Directional)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '02_shap_summary_dot.png'))
        plt.close()
        
        # 3. Top 3 Feature Dependence Plots
        # Calculate mean absolute SHAP values to find the top 3 most important features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:3]
        
        for i, idx in enumerate(top_indices):
            feature_name = X_train_scaled.columns[idx]
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(idx, shap_values, X_train_scaled, show=False)
            plt.title(f'SHAP Dependence Plot: {feature_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'0{3+i}_shap_dependence_{feature_name}.png'))
            plt.close()
            
    except Exception as e:
        print(f"Warning: Could not generate SHAP plots. Ensure 'shap' library is installed correctly. Error: {e}")

if __name__ == "__main__":
    main()
