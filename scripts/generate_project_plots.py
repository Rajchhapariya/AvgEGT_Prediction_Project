"""
Project: AvgEGT Prediction (Utility)
Script: generate_project_plots.py
Purpose: A lightweight plotting script. Unlike the exhaustive batch generator, this script only 
         generates the 10 core "Executive Summary" diagnostic plots (e.g. Correlation heatmaps, 
         Model Comparisons, and Error Distributions).
Outputs: Creates and saves 10 images into the `generated_project_plots/` directory.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# PHASE 1: CONFIGURATION & SETUP
# =====================================================================
DATA_PATH = os.path.join("..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("data", "raw", "AE_DATA_with_AvgEGT.csv")
OUTPUT_DIR = "generated_project_plots" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "AvgEGT"
EXCLUDED_COLS = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3",
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]

def main():
    """
    Main function to orchestrate data loading, cleaning, and plot generation.
    """
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}. Please ensure data is present.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # =====================================================================
    # PHASE 2: DATA CLEANING & PREPROCESSING
    # =====================================================================
    # Drop duplicates and any remaining missing values for robustness
    df = df.drop_duplicates().dropna()
    
    # Clean extreme outlier/impossible values (e.g., Temperature > 1000C)
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL] <= 1000]
    
    # Drop excluded columns to prevent data leakage
    drop_cols = [c for c in EXCLUDED_COLS if c in df.columns]
    df_clean = df.drop(columns=drop_cols)
    
    print(f"Data cleaned. Proceeding to generate plots in '{OUTPUT_DIR}' directory...")
    
    # =====================================================================
    # PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)
    # =====================================================================
    generate_eda_plots(df_clean)
    
    # =====================================================================
    # PHASE 4: DATA PREPARATION FOR MODELING
    # =====================================================================
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]
    
    # Train-test split (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    # =====================================================================
    # PHASE 5: MODEL EVALUATION DIAGNOSTICS
    # =====================================================================
    generate_model_evaluation_plots(X_train_scaled, X_test_scaled, y_train, y_test, X.columns)
    
    print(f"All plots generated successfully in the '{OUTPUT_DIR}' directory!")


def generate_eda_plots(df):
    """
    Generates and saves Exploratory Data Analysis (EDA) plots.
    """
    print("Generating EDA plots...")
    
    # Plot 1: Target Distribution (Spread and skewness of the AvgEGT)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_COL], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {TARGET_COL}')
    plt.xlabel(TARGET_COL)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'))
    plt.close()
    
    # Plot 2: Correlation Heatmap (Mathematical relationship between features)
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_correlation_heatmap.png'))
    plt.close()

    # Plot 3: Top Correlated Features vs Target
    # Extracts the top 4 features most correlated with the target and creates scatter plots.
    top_corr_features = corr_matrix[TARGET_COL].abs().sort_values(ascending=False).index[1:5]
    if len(top_corr_features) > 0:
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(top_corr_features, 1):
            plt.subplot(1, len(top_corr_features), i)
            sns.scatterplot(data=df, x=feature, y=TARGET_COL, alpha=0.5, color='coral')
            plt.title(f'{feature} vs {TARGET_COL}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '03_top_features_vs_target.png'))
        plt.close()


def generate_model_evaluation_plots(X_train, X_test, y_train, y_test, feature_names):
    """
    Trains models, compares them, and generates evaluation & diagnostic plots.
    """
    print("Training models and generating evaluation plots...")
    
    models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = []
    best_model_name = None
    best_r2 = -float('inf')
    best_model_instance = None
    best_y_pred = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({'Model': name, 'R2': r2, 'RMSE': rmse})
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_instance = model
            best_y_pred = y_pred

    results_df = pd.DataFrame(results)

    # Plot 4: Model Comparison (R2 Score)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=results_df, x='Model', y='R2', palette='viridis')
    plt.title('Model Comparison - R² Score')
    plt.ylabel('R² Score')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_model_comparison_r2.png'))
    plt.close()

    # Plot 5: Model Comparison (RMSE)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=results_df, x='Model', y='RMSE', palette='magma')
    plt.title('Model Comparison - RMSE')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_model_comparison_rmse.png'))
    plt.close()

    # Plot 6: Actual vs Predicted (Best Model)
    # Perfect predictions lie on the red dashed diagonal line.
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, best_y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted ({best_model_name})')
    plt.xlabel('Actual AvgEGT')
    plt.ylabel('Predicted AvgEGT')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_actual_vs_predicted.png'))
    plt.close()

    # Plot 7: Residuals vs Predicted
    # We want mistakes (residuals) scattered randomly around 0.
    residuals = y_test - best_y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(best_y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted AvgEGT')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_residuals_plot.png'))
    plt.close()

    # Plot 8: Feature Importance
    if hasattr(best_model_instance, 'feature_importances_'):
        importances = best_model_instance.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]

        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances[indices], y=sorted_features, palette='mako')
        plt.title('Feature Importances')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '08_feature_importance.png'))
        plt.close()

    # Plot 9: Error Distribution (Histogram of Residuals)
    # Checks if errors are normally distributed.
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color='salmon')
    plt.title('Distribution of Residuals (Errors)')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '09_error_distribution.png'))
    plt.close()
    
    # Plot 10: Binned Confusion Matrix
    # Converts regression values into bins to visualize classification-like hit-rates.
    try:
        bins = np.linspace(y_test.min(), y_test.max(), 5) # Divide into 4 quantiles
        y_test_binned = np.digitize(y_test, bins)
        y_pred_binned = np.digitize(best_y_pred, bins)
        cm = confusion_matrix(y_test_binned, y_pred_binned)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Binned Confusion Matrix (Regression to Bins)')
        plt.xlabel('Predicted Bin')
        plt.ylabel('Actual Bin')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '10_binned_confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Skipping binned confusion matrix due to error: {e}")

if __name__ == "__main__":
    main()
