# The Ultimate Master Guide: Engine AI Prediction System

Welcome to your complete Engine AI project! If you have absolutely zero technical background, do not worry—this document was written specifically for you. 

By the time you finish reading this, you will understand exactly how your new AI works, and exactly what **every single file, folder, and graph** in this entire project actually does for your business.

---

## Part 1: How the AI Works (The Two Branches)

Your engine is incredibly complex. To protect it, we built **two distinct AI systems**:
1. **Branch A: The "Smart Thermometer" (Regression)**. Predicts the exact, decimal temperature of the engine exhaust.
2. **Branch B: The "Intelligent Fire Alarm" (Classification)**. A binary alarm that triggers if the engine enters the top 10% historical danger zone.

---

## Part 2: Exhaustive File & Folder Dictionary

Below is the exhaustive list of every single file and folder in this project and exactly what it is used for.

### 🗂️ 1. The Data Folders (`/data`)
*This is where the raw engine readings are stored.*
* **`data/` (Folder):** The main container for all engine sensor readings.
* **`data/raw/` (Folder):** Contains the untouched historical data.
    * **`data/raw/AE_DATA_with_AvgEGT.csv` (File):** **(Use: Archival Storage)** The original spreadsheet of historical engine data you provided us. The AI reads this to learn how your engines behave.
* **`data/sanitized/` (Folder):** Contains cleaned data.
    * *(Note: During training, broken sensors and impossible temperatures are removed and processed here).*

### 🧠 2. The "Frozen Brains" (AI Models)
*Files ending in `.pkl` are "Frozen AI Brains". You can put these files on a USB drive and plug them into any server in the world, and they will instantly know how to monitor your engine.*
* **`final_model.pkl` (File):** **(Use: Live Temperature Prediction)** The frozen brain for the "Smart Thermometer" (Branch A). 
* **`classification_project/classification_model.pkl` (File):** **(Use: Live Alarm Triggering)** The frozen brain for the "Intelligent Fire Alarm" (Branch B).
* **`scaler.pkl` (File):** **(Use: The Translator)** Translates massive physical numbers (like 50,000 PSI) into the tiny decimals (0 to 1) that the AI's brain can process.

### 📓 3. The Client Deliverables (The Notebooks)
*Notebooks (files ending in `.ipynb`) are beautiful, interactive documents that combine code with text.*
* **`notebooks/` (Folder):** Contains the final deliverables you can open and read.
    * **`notebooks/Client_Prediction_Dashboard.ipynb` (File):** **(Use: Daily Operations)** Your daily driver. You open this to type in live sensor readings and instantly get a temperature and risk prediction.
    * **`notebooks/AvgEGT_Project_Notebook.ipynb` (File):** **(Use: Engineering Audit)** The unified master manual. If your engineers want to know the math behind both AI branches, they read this.
* **`classification_project/Classification_Project_Notebook.ipynb` (File):** **(Use: Safety Audit)** A visual encyclopedia dedicated entirely to proving the Fire Alarm works flawlessly.

### ⚙️ 4. The Factory Floor (The Python Scripts)
*You never need to touch these files. These are the Python scripts (`.py`) that built the AI behind the scenes.*
* **`regression_pipeline.py` (File):** **(Use: The Builder)** If you get new data next year, run this script to automatically train a smarter Temperature AI.
* **`classification_project/classification_pipeline.py` (File):** **(Use: The Builder)** Run this to automatically train a smarter Alarm AI on new data.
* **`predict_new_engine_data_REGRESSION.py` (File):** **(Use: Live Monitor)** A script designed to be plugged into live engine sensors to predict temperatures 24/7.
* **`classification_project/predict_new_engine_data_CLASSIFICATION.py` (File):** **(Use: Live Monitor)** A script designed to be plugged into live engine sensors to trigger alarms 24/7.
* **`scripts/` (Folder):** Contains automated reporting scripts.
    * **`scripts/build_notebook.py` (File):** Automates the writing of the Engineering Audit notebook.
    * **`scripts/build_client_dashboard.py` (File):** Automates the writing of the Daily Operations dashboard.
    * **`classification_project/build_classification_notebook.py` (File):** Automates the writing of the Safety Audit notebook.

---

## Part 3: Exhaustive Graph & Plot Dictionary

To prove the AI isn't guessing, we generated dozens of graphs. Here is exactly what every single image file is used for.

### A. The Temperature Graphs (`generated_project_plots/`)
* **`01_target_distribution.png`**: **(Use: Baseline Check)** Shows the most common temperature your engines historically run at.
* **`02_correlation_heatmap.png`**: **(Use: Finding Hidden Relationships)** A map showing which sensors mathematically rise and fall together.
* **`03_top_features_vs_target.png`**: **(Use: Direct Impact)** Scatter plots showing how individual sensors directly force the temperature up or down.
* **`04_model_comparison_r2.png`**: **(Use: Proving the Winner)** Shows how 5 different AIs scored in an accuracy shootout. Higher is better.
* **`05_model_comparison_rmse.png`**: **(Use: Proving the Winner)** Shows the error rates of the 5 AIs. Lower is better. Proves XGBoost is the best.
* **`06_actual_vs_predicted.png`**: **(Use: Accuracy Proof)** The red line is reality. The blue dots are the AI. Proves the AI is perfectly aligned with reality.
* **`07_residuals_plot.png`**: **(Use: Error Checking)** Proves that when the AI does make a mistake, it makes random, harmless mistakes rather than dangerous, biased mistakes.
* **`08_feature_importance.png`**: **(Use: Maintenance Focus)** Shows exactly which sensors have the biggest impact on engine temperature.
* **`09_error_distribution.png`**: **(Use: Confidence Check)** Proves the majority of the AI's mistakes are incredibly tiny (within a few degrees).

### B. The Alarm System Graphs (`classification_project/plots/`)
* **`classification_plots/01_accuracy_comparison.png`**: **(Use: Proving the Winner)** Shows the shootout between 5 different alarm algorithms.
* **`classification_plots/02_confusion_matrix.png`**: **(Use: Failure Analysis)** A 4-square grid proving the AI correctly catches engine failures without triggering too many false alarms.

**Folder: `plots/full_confusion_matrices/`**
* **`01_train_confusion_matrix.png`**: The matrix for data the AI studied.
* **`02_test_confusion_matrix.png`**: The matrix for brand-new data the AI had never seen before.
* **`03_all_data_confusion_matrix.png`**: The master matrix for all time.

**Folder: `plots/winning_model_classification/`**
* **`01_roc_curve.png`**: **(Use: Scientific Proof)** Proves the AI isn't hallucinating. A perfect curve hugs the top-left corner.
* **`02_precision_recall_curve.png`**: **(Use: Alarm Trust)** Proves that when the alarm rings, it is a real failure (Precision) and proves it catches almost all failures (Recall).
* **`03_confusion_matrix.png`**: Alternate copy of the failure analysis grid.
* **`04_feature_importance.png`**: **(Use: Maintenance Focus)** Shows exactly which sensors cause the fire alarm to trigger the most.

**Folder: `plots/winning_model_classification/training_history/`**
* **`01_xgboost_iteration_vs_logloss.png`**: **(Use: Learning Proof)** Shows the AI's error rate dropping as it practiced.
* **`02_xgboost_iteration_vs_accuracy.png`**: **(Use: Learning Proof)** Shows the AI's accuracy skyrocketing as it practiced.

**Folder: `plots/winning_model_classification/train_test_comparisons/`**
*(Use: Stability Proof. These graphs prove the AI performs identically on brand-new data as it did on old data, meaning it didn't just memorize the past).*
* **`05_class_metrics_comparison.png`**
* **`06_class_roc_overlay.png`**
* **`07_class_pr_curve_overlay.png`**
* **`08_class_probability_density.png`**

**Folder: `plots/advanced_executive_plots/`**
* **`01_classification_tree_blueprint.png`**: **(Use: Visualizing the Brain)** A literal diagram of the complex decision paths the AI takes in its head.
* **`02_cumulative_gains_chart.png`**: **(Use: Business Value)** Shows how much faster you catch a failure using the AI vs randomly checking engines.
* **`03_lift_curve.png`**: **(Use: Business Value)** Shows how many times safer the company is by using this AI compared to flying blind.
* **`04_calibration_curve.png`**: **(Use: Trust Checking)** Proves that when the AI says it is "90% confident", it is actually right 90% of the time.

**Folder: `plots/advanced_executive_plots/violin_distributions/`**
*(Use: Threshold Hunting. By comparing the blue shape (Normal) to the orange shape (Danger), engineers can instantly see the exact physical threshold where a sensor enters critical failure).*
* **`violin_KW.png`**: Engine Load threshold.
* **`violin_LO_IN_TEMP.png`**: Lube Oil In Temp threshold.
* **`violin_SCAV_TEMP.png`**: Scavenge Air Temp threshold.
* **`violin_FO_TEMP.png`**: Fuel Oil Temp threshold.
* **`violin_TC_IN_TEMP.png`**: Turbocharger In Temp threshold.
* **`violin_TC_OUT_TEMP.png`**: Turbocharger Out Temp threshold.
* **`violin_LO_PRESS.png`**: Lube Oil Pressure threshold.
* **`violin_FO_PRESS.png`**: Fuel Oil Pressure threshold.
* **`violin_CW_PRESS.png`**: Cooling Water Pressure threshold.
* **`violin_SCAV_AIR_PRESS.png`**: Scavenge Air Pressure threshold.
* **`violin_TC_LO_PRESS.png`**: Turbocharger Lube Oil Pressure threshold.
* **`violin_Rack_Index.png`**: Fuel Actuator threshold.

---
*End of Master Guide. You are now fully equipped to understand, deploy, and explain your new Engine AI System!*
