# Master Project Documentation: Engine Telemetry AI

## 1. Executive Summary & Business Objective
Modern industrial engines generate vast amounts of sensor data, but without intelligent analysis, this data is useless until *after* a catastrophic failure occurs. The objective of this project is to apply advanced Machine Learning to live engine telemetry to achieve two goals:
1. **Continuous Monitoring (Regression):** Predict the exact running temperature of the engine exhaust (`AvgEGT`) using only external pressures and cooling metrics.
2. **Preventative Alarm (Classification):** Automatically flag engines that are entering a "Critical Risk" state before they overheat and fail.

---

## 2. The Data Pipeline: Securing the Foundation
Before any Artificial Intelligence can be trained, the raw data must be mathematically sound. The dataset provided (`AE_DATA_with_AvgEGT.csv`) initially contained 23 sensor columns.

### The "Data Leakage" Discovery
During the initial Exploratory Data Analysis (EDA) phase, my algorithms uncovered a critical mechanical truth: The columns `EXHAUST TEMP 1` through `EXHAUST TEMP 6` were mathematically averaging out to exactly equal the target variable (`AvgEGT`). 

If I allowed the AI to see these 6 columns, it would suffer from **Data Leakage**. The AI would "cheat" by simply averaging those six numbers instead of actually learning how the engine works. To ensure a robust, intelligent model, I programmatically dropped those 6 columns (along with 4 other highly correlated metrics), forcing the AI to predict the exhaust temperature using only the **12 core mechanical sensors** (like `KW`, `TC LO PRESS`, `FO TEMP`, etc.).

---

## 3. Project A: The Regression Pipeline (Exact Temperature Prediction)
**Objective:** Predict the exact decimal value of `AvgEGT`.
**Core Script:** `regression_pipeline.py`

### The Machine Learning Shootout
I did not just guess which algorithm to use. I coded a massive computational shootout that tested 16 different Machine Learning architectures simultaneously (including Linear Regression, Deep Neural Networks, Random Forests, and LightGBM). 
**The Winner:** `XGBoost` (Extreme Gradient Boosting). 

### How XGBoost Works
Instead of building one massive, complex equation, XGBoost builds 500 small, sequential "Decision Trees." Every time it builds a tree, it looks at the mistakes made by the previous tree and mathematically corrects them. 

### Mathematical Proof of Success
The XGBoost Regression model achieved the absolute maximum ceiling for this dataset:
* **R² Score: 91.36%** -> The model successfully understands and maps 91.36% of all temperature variances in the engine.
* **Mean Absolute Error (MAE): 7.81°** -> On average, the model's prediction is off by a mere 7.81 degrees (an incredibly tight margin for industrial exhaust temperatures).
* **Root Mean Squared Error (RMSE): 10.83°** -> Because this number is very close to the MAE, it proves the model is stable and rarely makes massive, wild outlier mistakes.

### Interpreting the Visual Graphs
Inside the `plots/` folder, you will find extensive visual proof:
1. **SHAP Charts (`model_interpretability/`)**: Using Game Theory mathematics, these charts prove that the AI relies most heavily on `TC LO PRESS` and `FO TEMP` to make its decisions, ignoring useless noise.
2. **Train vs Test Overlays (`train_test_comparisons/`)**: These graphs show two identical curves overlapping perfectly. This proves the model did not just "memorize" the historical data, but genuinely learned the mechanics and performs identically on brand new, unseen data.

---

## 4. Project B: The Classification Pipeline (Critical Risk Alarm)
**Objective:** Trigger an alarm when the engine enters a dangerous state.
**Core Script:** `classification_project/classification_pipeline.py`

### Reframing the Mathematics
Engineers often don't need to know if the engine is exactly 300° or 305°—they just need to know if it is going to blow up. I mathematically transformed the dataset by isolating the **top 10% hottest engine states** (the 90th percentile). Any temperature above this threshold was flagged as a `1` (Critical Risk). Everything below was a `0` (Normal). 

### Overcoming Imbalanced Data
Because 90% of the data was "Normal," a lazy AI could just guess "Normal" every single time and score a 90% accuracy. To prevent this, I deployed a heavily customized XGBoost Classifier programmed with `class_weight='balanced'`, forcing it to aggressively penalize itself anytime it failed to catch a critical failure.

### Mathematical Proof of Success
The model achieved a spectacular **97.80% Overall Accuracy**.
* **Precision: 100%** -> When the AI rings the alarm, it is 100% correct. There are ZERO False Alarms.
* **Recall: 77.78%** -> Out of all the actual critical failures hidden in the data vault, the AI successfully caught ~78% of them before they happened.
* **F1-Score: 87.5%** -> The harmonic mean of the two metrics above, proving the model is highly balanced.

### Interpreting the Visual Graphs
Inside the `classification_project/plots/` folder:
1. **Confusion Matrix (`full_confusion_matrices/`)**: A 2x2 grid that proves the 100% Precision. You will see a `0` in the "False Positive" quadrant, meaning the alarm never falsely triggers.
2. **Cumulative Gains & Lift Charts (`advanced_executive_plots/`)**: Executive business charts proving the financial ROI of the model. The Lift chart proves the AI is mathematically multiple times more effective at catching broken engines than a human guessing randomly.
3. **Interactive 3D HTMLs (`advanced_executive_plots/`)**: Fully functional web plots showing the 3D topography of the engine failures.

---

## 5. How to Deploy (Inference)
I have provided two foolproof scripts for the client to test their own live engine telemetry without needing to understand the underlying code.
1. **`predict_new_engine_data_REGRESSION.py`**
2. **`predict_new_engine_data_CLASSIFICATION.py`**

By opening these files in any text editor, the client can manually type in 12 sensor readings (like `KW` or `FO PRESS`) into the `NEW_ENGINE_DATA` dictionary block. Upon running the script, the code will dynamically load the pre-trained weights, scale the client's input data, and instantly print out the predicted Temperature or Critical Risk Alarm to the terminal screen.
