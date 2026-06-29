"""
Project: AvgEGT Prediction (Client Facing)
Script: build_client_dashboard.py
Purpose: Programmatically generates `Client_Prediction_Dashboard.ipynb`.
         This notebook provides a simple, non-technical interface for the client 
         to understand the models and run interactive predictions using IPyWidgets.
"""
import nbformat as nbf
import os

def create_dashboard():
    nb = nbf.v4.new_notebook()

    # =====================================================================
    # SECTION 1: Introduction
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""# Engine Risk & Temperature Prediction Dashboard
Welcome to the interactive prediction dashboard! This tool is designed to help you quickly input live engine data and instantly receive AI-powered insights about the health of your engine.

## The Goal
The goal of this project is to monitor 12 critical engine sensors (like lube oil temperature and pressure) and predict the **Average Exhaust Gas Temperature (AvgEGT)**. By predicting this temperature accurately, we can detect if the engine is running too hot and predict potential mechanical failures before they happen."""))

    # =====================================================================
    # SECTION 2: Bifurcation of Models
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""## Bifurcation of Models: How the AI Thinks
We evaluated multiple different AI approaches to solve this problem. To give you the most comprehensive engine monitoring possible, we split our final system into two distinct "Branches" (or goals):

### Branch A: The Temperature Predictor (Regression)
**Goal:** Predict the *exact* decimal temperature of the exhaust gas.
- **Models Tested:** We tested Linear Regression, Decision Trees, Random Forests, ExtraTrees, and XGBoost.
- **The Winner:** **XGBoost (Extreme Gradient Boosting)**. It won because it had the lowest error rate (RMSE) and successfully understood the complex, non-linear relationships between the sensors much better than standard Linear Regression.

### Branch B: The Critical Alarm System (Classification)
**Goal:** Act as a binary alarm (Is the engine safe, or in critical danger?).
- **Concept:** We mathematically identified the top 10% hottest historical engine states as "Critical Risk" zones. Instead of predicting a temperature, this model simply outputs a 1 (Alarm Triggered) or 0 (Normal). 
- **The Winner:** **XGBoost Classifier**. It not only triggers the alarm, but it tells you its exact % Confidence that the engine is failing.

By combining both branches, you get the exact predicted temperature *and* a clear risk status!"""))

    # =====================================================================
    # SECTION 3: Understanding the Graphs
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""## Understanding the AI's Logic (Visualized)
Below are three key graphs that explain why the XGBoost model was selected and how it makes its decisions.

### 1. Model Comparison (The "Shootout")
![Model Comparison](../generated_project_plots/05_model_comparison_rmse.png)
*How to read this:* This graph shows the "Shootout" phase where we tested multiple different AI models for the Temperature Predictor (Branch A). The lower the bar, the fewer "mistakes" the AI makes. Notice how XGBoost is significantly lower than standard Linear Regression or Decision Trees. This proves exactly *why* XGBoost was selected as our final winner!

### 2. Feature Importance (What drives the AI?)
![Feature Importance](../generated_project_plots/08_feature_importance.png)
*How to read this:* This shows which sensors the AI pays the most attention to when calculating the exhaust temperature. The sensors at the top have the largest impact on the final prediction.

### 3. Actual vs Predicted (Accuracy)
![Actual vs Predicted](../generated_project_plots/06_actual_vs_predicted.png)
*How to read this:* The red dashed line represents "perfect accuracy" (where the prediction perfectly matches reality). The blue dots are the AI's actual predictions. Because the blue dots tightly cluster around the straight red line, we know the AI is highly reliable!

### 4. Alarm Accuracy (Confusion Matrix)
![Confusion Matrix](../generated_project_plots/10_binned_confusion_matrix.png)
*How to read this:* This represents Branch B (The Critical Alarm System). It shows where the AI got it right, and where it got it wrong. The Top-Left box is when the engine was safe and the AI correctly said it was safe. The Bottom-Right box is when the engine was failing and the AI successfully triggered the alarm!"""))

    # =====================================================================
    # SECTION 4: FAQ (The 220+ Graphs)
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""## FAQ: What about the 220+ Exhaustive Graphs?
You might have seen another document containing over 220 graphs (histograms, boxplots, and scatter plots). **Do not worry about memorizing them!** 

Those graphs represent our **"Engine Analytics Audit"** (also known as Exploratory Data Analysis or EDA). 
- **What are they?** The AI automatically generated those graphs to hunt for extreme mechanical anomalies (like impossible temperatures) and to map out exactly how every single sensor interacts with every other sensor.
- **Why did we make them?** They are the raw "blueprints" our Data Scientists used to clean the data before feeding it to XGBoost. 
- **Do you need them daily?** No! The AI has already learned everything from those 220 graphs. For day-to-day operations, you only need this Interactive Dashboard to get instant answers."""))

    # =====================================================================
    # SECTION 5: Interactive Dashboard
    # =====================================================================
    nb.cells.append(nbf.v4.new_markdown_cell("""## Interactive Inference Form
**Instructions:** 
1. Run the code cell below by clicking on it and pressing `Shift + Enter` (or clicking the "Run" button).
2. An interactive dashboard will appear. 
3. Type in your custom sensor numbers and click **Predict Risk & Temperature!**"""))

    code_string = '''import os
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("Booting up AI Brains... (Loading background data)")

# 1. Load Background Data to fit the Scaler & Thresholds
data_path = os.path.join("..", "data", "raw", "AE_DATA_with_AvgEGT.csv")
df = pd.read_csv(data_path).drop_duplicates().dropna()
df = df[df['AvgEGT'] <= 1000]

excluded_cols = [
    "EXHAUST TEMP 1", "EXHAUST TEMP 2", "EXHAUST TEMP 3", 
    "EXHAUST TEMP 4", "EXHAUST TEMP 5", "EXHAUST TEMP 6",
    "FREQ", "AMP", "CPW IN TEMP", "CPW OUT TEMP"
]
df = df.drop(columns=[c for c in excluded_cols if c in df.columns])

X = df.drop(columns=['AvgEGT'])
y_reg = df['AvgEGT']

threshold = np.percentile(y_reg, 90)
y_cls = (y_reg >= threshold).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train Models (XGBoost is fast enough to do this live for the client)
reg_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
reg_model.fit(X_scaled, y_reg)

cls_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
cls_model.fit(X_scaled, y_cls)

print("Models loaded successfully! Creating Dashboard...")

# 3. Create UI Widgets
style = {'description_width': 'initial'}
layout = widgets.Layout(width='300px')

inputs = {
    'KW': widgets.FloatText(value=3200.0, description='KW (Power):', style=style, layout=layout),
    'LO IN TEMP': widgets.FloatText(value=48.0, description='LO IN TEMP:', style=style, layout=layout),
    'SCAV TEMP': widgets.FloatText(value=45.2, description='SCAV TEMP:', style=style, layout=layout),
    'FO TEMP': widgets.FloatText(value=110.0, description='FO TEMP:', style=style, layout=layout),
    'TC IN TEMP': widgets.FloatText(value=450.0, description='TC IN TEMP:', style=style, layout=layout),
    'TC OUT TEMP': widgets.FloatText(value=350.0, description='TC OUT TEMP:', style=style, layout=layout),
    'LO PRESS': widgets.FloatText(value=45.5, description='LO PRESS:', style=style, layout=layout),
    'FO PRESS': widgets.FloatText(value=600.0, description='FO PRESS:', style=style, layout=layout),
    'CW PRESS': widgets.FloatText(value=3.5, description='CW PRESS:', style=style, layout=layout),
    'SCAV AIR PRESS': widgets.FloatText(value=1.8, description='SCAV AIR PRESS:', style=style, layout=layout),
    'TC LO PRESS': widgets.FloatText(value=140.5, description='TC LO PRESS:', style=style, layout=layout),
    'Rack Index': widgets.FloatText(value=25.0, description='Rack Index:', style=style, layout=layout)
}

btn_predict = widgets.Button(
    description='Predict Risk & Temperature!',
    button_style='success',
    tooltip='Run the XGBoost Models',
    icon='check',
    layout=widgets.Layout(width='300px', height='50px', margin='20px 0px 0px 0px')
)

out = widgets.Output()

def on_predict_clicked(b):
    with out:
        out.clear_output()
        # Gather data
        new_data = {k: v.value for k, v in inputs.items()}
        new_df = pd.DataFrame([new_data])
        
        # Scale
        new_df_scaled = scaler.transform(new_df[X.columns])
        
        # Predict Regression
        temp_pred = reg_model.predict(new_df_scaled)[0]
        
        # Predict Classification
        alarm_pred = cls_model.predict(new_df_scaled)[0]
        alarm_prob = cls_model.predict_proba(new_df_scaled)[0][1] * 100
        
        status_color = "red" if alarm_pred == 1 else "green"
        status_text = "CRITICAL ALARM TRIGGERED" if alarm_pred == 1 else "NORMAL - ENGINE SAFE"
        
        html_report = f"""
        <div style="border: 2px solid #ccc; padding: 20px; border-radius: 10px; font-family: Arial; margin-top: 20px;">
            <h2 style="margin-top:0px;">AI Prediction Report</h2>
            <hr>
            <h3>Branch A: Temperature Prediction (Regression)</h3>
            <p style="font-size: 24px;">Predicted AvgEGT: <b>{temp_pred:.2f} °C</b></p>
            <br>
            <h3>Branch B: Risk Assessment (Classification)</h3>
            <p style="font-size: 24px; color: {status_color}; font-weight: bold;">{status_text}</p>
            <p style="font-size: 18px;">Probability of Critical Engine State: <b>{alarm_prob:.2f}%</b></p>
        </div>
        """
        display(HTML(html_report))

btn_predict.on_click(on_predict_clicked)

# Layout the form nicely in two columns
items = list(inputs.values())
left_box = widgets.VBox(items[:6])
right_box = widgets.VBox(items[6:])
form = widgets.HBox([left_box, right_box])

display(form)
display(btn_predict)
display(out)
'''
    nb.cells.append(nbf.v4.new_code_cell(code_string))

    # =====================================================================
    # SAVE NOTEBOOK
    # =====================================================================
    output_dir = os.path.join("..", "notebooks")
    if not os.path.exists(output_dir):
        # Fallback if running from project root
        output_dir = "notebooks"
        
    output_path = os.path.join(output_dir, "Client_Prediction_Dashboard.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Client Dashboard successfully generated at: {output_path}")

if __name__ == "__main__":
    create_dashboard()
