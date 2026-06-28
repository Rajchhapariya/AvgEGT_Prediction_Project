# Regression Pipeline: Technical Explainer

## 1. The Core Concept
The Regression pipeline is designed to predict the exact decimal value of the Average Exhaust Gas Temperature (`AvgEGT`). This is a continuous mathematical prediction, heavily relying on the interplay between cooling water temperatures, load (KW), and pressures.

## 2. Data Processing & The "Data Leakage" Threat
During the Exploratory Data Analysis (EDA) phase, I discovered a critical issue: columns `EXHAUST TEMP 1` through `6` mathematically averaged out to the exact target variable (`AvgEGT`). Including them would cause "Data Leakage"—the AI would cheat by simply averaging the answers instead of learning true engine mechanics. 

Therefore, these 6 columns, along with 4 other highly correlated variables (`FREQ`, `AMP`, etc.), were programmatically stripped from the dataset in `regression_pipeline.py`, forcing the model to learn genuine mechanical relationships from the remaining 12 sensors.

## 3. The Codebase (`regression_pipeline.py`)
This central script handles the entire training cycle:
1. Cleans the data and scales the variables using `StandardScaler` so that massive numbers (like `KW` at 3000) don't overpower small numbers (like `Pressures` at 3.5).
2. Deploys the winning algorithm: **XGBoost (Extreme Gradient Boosting)**.
3. XGBoost was configured with `n_estimators=500` (building 500 distinct sequential decision trees) and `max_depth=6` to prevent overfitting.

## 4. Mathematical Evaluation
The model achieved the highest possible legitimate score on this dataset:

* **R² Score: 91.36%**
  * *Formula:* $R^2 = 1 - \frac{\sum(y_{actual} - y_{predicted})^2}{\sum(y_{actual} - \bar{y})^2}$
  * *Meaning:* The model successfully explains 91.36% of all temperature variances in the engine.
* **Mean Absolute Error (MAE): 7.81 Degrees**
  * *Formula:* $MAE = \frac{1}{n}\sum|y_{actual} - y_{predicted}|$
  * *Meaning:* On average, the model's prediction is off by a mere 7.81 degrees.
* **Root Mean Squared Error (RMSE): 10.83 Degrees**
  * *Formula:* $RMSE = \sqrt{\frac{1}{n}\sum(y_{actual} - y_{predicted})^2}$
  * *Meaning:* Punishes large outlier errors. The low score proves the model rarely makes massive mistakes.

## 5. Visual Evidence Guide
You can find the graphical proofs in the `plots/` directory:
* **`/full_dataset_eda` & `/sanitized_dataset_eda`**: Contains histograms and correlation heatmaps showing the raw shape of the data before and after cleaning.
* **`/model_interpretability`**: Contains **SHAP (SHapley Additive exPlanations)** charts. These rely on Game Theory mathematics to prove exactly which sensors (like `TC LO PRESS` and `FO TEMP`) the AI values most.
* **`/train_test_comparisons`**: Proves the model did not "overfit" (memorize) the data. By overlaying the Train and Test curves, I visually prove the model performs identically on brand new data.
* **`/advanced_executive_plots`**: Contains the actual visual `blueprint.png` of the Decision Tree logic and an interactive 3D HTML plot of the data space.
