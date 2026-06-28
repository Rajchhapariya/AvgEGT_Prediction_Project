# Engine Temperature & Risk Prediction - Client Runbook

## 1. Executive Summary
This project utilizes state-of-the-art Machine Learning to monitor engine sensor telemetry. It delivers two core AI solutions:
1. **The Regression Model**: Predicts the exact continuous temperature (`AvgEGT`) of the engine based on live sensor readings with **91.36% accuracy**.
2. **The Classification Model**: Acts as an early-warning alarm system. It mathematically analyzes the data to flag engines running in the top 10% of critical risk, achieving a massive **97.80% accuracy** with zero false alarms.

---

## 2. Setup & Installation
This project requires **Python 3.11** (or compatible 3.x versions). To run this project on a new machine safely, it is highly recommended to use a Virtual Environment.

1. Open your terminal or command prompt.
2. Navigate to the root directory of this project.
3. **Create a Virtual Environment**:
   * *Windows:* `python -m venv venv`
   * *Mac/Linux:* `python3 -m venv venv`
4. **Activate the Virtual Environment**:
   * *Windows:* `.\venv\Scripts\activate`
   * *Mac/Linux:* `source venv/bin/activate`
5. **Install Dependencies**:
   Run the following command to install all required AI libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. How to Test & Predict New Engine Data
I have provided two simple, standalone inference scripts that allow you to plug in your own sensor numbers and instantly receive AI predictions.

### Testing the Exact Temperature Predictor
1. Open `predict_new_engine_data_REGRESSION.py` in any text editor.
2. Locate the `NEW_ENGINE_DATA` dictionary block around Line 40.
3. Modify the 12 sensor values (KW, FO TEMP, TC LO PRESS, etc.) to simulate any engine state you desire.
4. Save the file and run it in your terminal:
   ```bash
   python predict_new_engine_data_REGRESSION.py
   ```
5. The terminal will instantly print out the exact predicted Exhaust Gas Temperature (`AvgEGT`).

### Testing the Critical Failure Alarm
1. Navigate to the classification folder: `cd classification_project`
2. Open `predict_new_engine_data_CLASSIFICATION.py` in a text editor.
3. Modify the `NEW_ENGINE_DATA` sensor values. Try cranking up the temperatures and lowering the pressures to simulate a failing engine!
4. Save the file and run it in your terminal:
   ```bash
   python predict_new_engine_data_CLASSIFICATION.py
   ```
5. The terminal will instantly trigger a `CRITICAL RISK` alarm or a `NORMAL` status, along with the exact % confidence that the engine will fail.

---

## 4. Navigating the Analytics
This project contains over 220 visual charts and scoreboards proving the validity of the models.
* **Scoreboards**: You can find Excel-ready CSV files containing the final ranked scores of all 16 tested AI models in `plots/regression_model_comparison_scoreboard.csv` and `classification_project/plots/classification_model_comparison_scoreboard.csv`.
* **Interactive 3D Plots**: Located inside `plots/advanced_executive_plots/`. Open the `.html` files in any web browser to click, drag, and rotate the 3D data topologies.
* **Visual Evidence**: All other folders (like `training_history`, `model_interpretability`, and `train_test_comparisons`) contain detailed graphical proofs of the model's accuracy.
