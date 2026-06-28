import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# AvgEGT Prediction: Post-Cleaning Full EDA
This notebook is explicitly designed to perform an exhaustive Exploratory Data Analysis (EDA) on **all 23 original columns** of the dataset, but **only after** the rows have been properly cleaned (removing duplicates, blank cells, and extreme outliers)."""))

    # Imports
    nb.cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline"""))

    # Data Loading and Cleaning
    nb.cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading & Row Cleaning
Here we load the raw dataset and perform row-level cleaning (dropping duplicates and blanks). Notice that we are **not** dropping the excluded feature columns yet."""))

    nb.cells.append(nbf.v4.new_code_cell("""DATA_PATH = os.path.join("..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
TARGET_COL = "AvgEGT"

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Raw Data Shape: {df.shape}")

# Drop duplicates and blank cells
df_cleaned_all_cols = df.drop_duplicates().dropna()

# Remove extreme outliers in Target
if TARGET_COL in df_cleaned_all_cols.columns:
    df_cleaned_all_cols = df_cleaned_all_cols[df_cleaned_all_cols[TARGET_COL] <= 1000]

print(f"Cleaned Data Shape (Still has all 23 columns): {df_cleaned_all_cols.shape}")
df_cleaned_all_cols.head()"""))

    # Exhaustive Plot Generation Section
    nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Exhaustive Plot Generation (Inline)
This section programmatically generates the Distribution, Boxplot, and Scatter Plot (against the target) for every single column in the dataset using the perfectly clean rows."""))

    nb.cells.append(nbf.v4.new_code_cell("""def generate_feature_loops_inline(df, prefix=""):
    features = [c for c in df.columns if c != TARGET_COL]
    
    print(f"Generating plots for {len(features)} features...")
    
    for i, feature in enumerate(features):
        # 1. Feature Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True, color='teal')
        plt.title(f'[{prefix.upper()}] Distribution of {feature}')
        plt.tight_layout()
        plt.show()
        
        # 2. Boxplot
        plt.figure(figsize=(8, 2))
        sns.boxplot(x=df[feature], color='orange')
        plt.title(f'[{prefix.upper()}] Boxplot of {feature}')
        plt.tight_layout()
        plt.show()
        
        # 3. Scatter vs Target
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=feature, y=TARGET_COL, alpha=0.6, color='navy')
        plt.title(f'[{prefix.upper()}] {feature} vs {TARGET_COL}')
        plt.tight_layout()
        plt.show()

generate_feature_loops_inline(df_cleaned_all_cols, prefix="Cleaned_All_Cols")
print("All plots rendered inline successfully!")"""))

    # Append counter script
    nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Image Counter
Run this cell to count how many images were rendered in this notebook."""))

    nb.cells.append(nbf.v4.new_code_cell("""import json

notebook_filename = "AvgEGT_PostCleaning_EDA_Notebook.ipynb"

try:
    with open(notebook_filename, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)

    image_count = 0
    for cell in notebook_data.get("cells", []):
        for output in cell.get("outputs", []):
            if "data" in output and "image/png" in output["data"]:
                image_count += 1

    print(f"✅ Total graphs generated and rendered in this notebook: {image_count}")
except FileNotFoundError:
    print(f"Could not find {notebook_filename}. Please save the notebook first or check the filename!")"""))

    output_path = os.path.join("..", "notebooks", "AvgEGT_PostCleaning_EDA_Notebook.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
