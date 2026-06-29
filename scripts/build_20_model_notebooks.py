import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import glob

# Ensure output directory exists
output_dir = os.path.join("..", "notebooks", "individual_models")
os.makedirs(output_dir, exist_ok=True)

# Helper to read python file
def read_py_file(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    # Patch the relative data path for the notebook execution context
    code = code.replace("'../data/raw/AE_DATA_with_AvgEGT.csv'", "'../../data/raw/AE_DATA_with_AvgEGT.csv'")
    code = code.replace('"../data/raw/AE_DATA_with_AvgEGT.csv"', '"../../data/raw/AE_DATA_with_AvgEGT.csv"')
    return code

# Generate Regression Notebooks
regression_models = glob.glob(os.path.join("..", "models", "*.py"))

for model_file in regression_models:
    model_name = os.path.basename(model_file).replace(".py", "")
    nb = new_notebook()
    
    # Intro
    nb.cells.append(new_markdown_cell(f"# Isolated Model Analysis: {model_name.upper()}\n\nThis notebook is dedicated entirely to deeply analyzing the {model_name} regression model (The 'Smart Thermometer'). It generates 7 highly specific diagnostic graphs to prove the model's accuracy, check for biases, and find exactly where it makes mistakes."))
    
    # Imports & Training
    base_code = read_py_file(model_file)
    nb.cells.append(new_code_cell(base_code))
    
    # Plotting Imports
    nb.cells.append(new_code_cell("import matplotlib.pyplot as plt\nimport seaborn as sns\nimport scipy.stats as stats\nimport numpy as np\nimport pandas as pd"))
    
    # 1. Actual vs Predicted Scatter
    nb.cells.append(new_markdown_cell("### 1. Actual vs Predicted (Scatter Plot)\n**(Use: Accuracy Proof)** The red line represents perfect reality. The blue dots are the AI's predictions. The closer the dots hug the red line, the more perfectly accurate the AI is."))
    nb.cells.append(new_code_cell("plt.figure(figsize=(10,6))\nplt.scatter(y_test, y_pred, alpha=0.5, color='blue')\nplt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\nplt.xlabel('Actual Temperature')\nplt.ylabel('Predicted Temperature')\nplt.title('Actual vs Predicted')\nplt.grid(True)\nplt.show()"))
    
    # 2. Hexbin Density
    nb.cells.append(new_markdown_cell("### 2. Actual vs Predicted (Hexbin Density)\n**(Use: Density Proof)** Similar to the scatter plot, but colors show density. It proves where the vast majority of predictions land, rather than just showing outliers."))
    nb.cells.append(new_code_cell("plt.figure(figsize=(10,6))\nhb = plt.hexbin(y_test, y_pred, gridsize=30, cmap='Blues', mincnt=1)\nplt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\nplt.colorbar(hb, label='Number of Predictions')\nplt.xlabel('Actual Temperature')\nplt.ylabel('Predicted Temperature')\nplt.title('Hexbin Density: Actual vs Predicted')\nplt.show()"))
    
    # 3. Residuals vs Predicted
    nb.cells.append(new_markdown_cell("### 3. Residuals vs Predicted (Scatter)\n**(Use: Consistency Check - Homoscedasticity)** This checks if the AI makes larger mistakes when temperatures get hotter. We want the blue dots to look like a random cloud, not a funnel."))
    nb.cells.append(new_code_cell("residuals = y_test - y_pred\nplt.figure(figsize=(10,6))\nplt.scatter(y_pred, residuals, alpha=0.5)\nplt.axhline(0, color='red', linestyle='--')\nplt.xlabel('Predicted Temperature')\nplt.ylabel('Residual (Error)')\nplt.title('Residuals vs Predicted')\nplt.grid(True)\nplt.show()"))
    
    # 4. Residuals Distribution
    nb.cells.append(new_markdown_cell("### 4. Residuals Distribution (Histogram)\n**(Use: Bias Check)** This proves the mistakes the AI makes are completely random. We want a perfect bell curve centered at zero."))
    nb.cells.append(new_code_cell("plt.figure(figsize=(10,6))\nsns.histplot(residuals, kde=True, color='purple')\nplt.xlabel('Residual (Error in Degrees)')\nplt.title('Error Distribution')\nplt.grid(True)\nplt.show()"))
    
    # 5. Q-Q Plot
    nb.cells.append(new_markdown_cell("### 5. Q-Q Plot of Residuals\n**(Use: Hardcore Statistics Proof)** Another visual proof of a perfect bell curve. If the blue dots stick to the red line, the AI's mistakes follow a perfect normal distribution without extreme biases."))
    nb.cells.append(new_code_cell("plt.figure(figsize=(8,8))\nstats.probplot(residuals, dist='norm', plot=plt)\nplt.title('Q-Q Plot of Residuals')\nplt.grid(True)\nplt.show()"))
    
    # 6. Absolute Error Boxplot
    nb.cells.append(new_markdown_cell("### 6. Absolute Error Boxplot\n**(Use: Error Sizing)** Shows the exact range of mistakes. The box contains 50% of all predictions. It shows you exactly how tight the AI's accuracy window is."))
    nb.cells.append(new_code_cell("abs_error = np.abs(residuals)\nplt.figure(figsize=(10,4))\nsns.boxplot(x=abs_error, color='orange')\nplt.xlabel('Absolute Error (Degrees)')\nplt.title('Absolute Error Range')\nplt.grid(True)\nplt.show()"))
    
    # 7. Feature Importance
    nb.cells.append(new_markdown_cell("### 7. Feature Importance / Coefficients\n**(Use: Maintenance Focus)** Shows exactly which sensors this specific AI pays the most attention to when calculating temperature. *(Note: Not all mathematical models support this graph).*"))
    nb.cells.append(new_code_cell("try:\n    if hasattr(model, 'feature_importances_'):\n        imp = model.feature_importances_\n    elif hasattr(model, 'coef_'):\n        imp = np.abs(model.coef_)\n    else:\n        raise ValueError('No feature importances found for this model.')\n    \n    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': imp}).sort_values('Importance', ascending=False)\n    plt.figure(figsize=(12,6))\n    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')\n    plt.title(f'{model_name} Feature Importance')\n    plt.show()\nexcept Exception as e:\n    print(f'Cannot plot feature importance: {e}')"))
    
    # Save notebook
    out_path = os.path.join(output_dir, f"{model_name}_analysis.ipynb")
    with open(out_path, 'w') as f:
        nbformat.write(nb, f)
        
print(f"Generated {len(regression_models)} regression notebooks.")

# Generate Classification Notebooks
classification_models = glob.glob(os.path.join("..", "classification_project", "models", "*.py"))

for model_file in classification_models:
    model_name = os.path.basename(model_file).replace(".py", "")
    nb = new_notebook()
    
    # Intro
    nb.cells.append(new_markdown_cell(f"# Isolated Model Analysis: {model_name.upper()}\n\nThis notebook is dedicated entirely to deeply analyzing the {model_name} classification model (The 'Intelligent Fire Alarm'). It generates 9 highly specific diagnostic graphs to prove the alarm triggers exactly when it should without crying wolf."))
    
    # Imports & Training
    base_code = read_py_file(model_file)
    nb.cells.append(new_code_cell(base_code))
    
    # Plotting Imports
    nb.cells.append(new_code_cell("import matplotlib.pyplot as plt\nimport seaborn as sns\nimport numpy as np\nimport pandas as pd\nfrom sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, brier_score_loss\nfrom sklearn.calibration import calibration_curve"))
    
    # Setup Predict Proba
    nb.cells.append(new_code_cell("try:\n    y_prob = model.predict_proba(X_test_s)[:, 1]\nexcept:\n    if hasattr(model, 'decision_function'):\n        y_prob = model.decision_function(X_test_s)\n        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())\n    else:\n        y_prob = y_pred"))
    
    # 1. Raw Confusion Matrix
    nb.cells.append(new_markdown_cell("### 1. Raw Confusion Matrix (Heatmap)\n**(Use: Exact Failure Count)** A grid showing exactly how many times the AI was perfectly correct vs. how many False Alarms or Missed Dangers occurred."))
    nb.cells.append(new_code_cell("cm = confusion_matrix(y_test, y_pred)\nplt.figure(figsize=(6,5))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Normal', 'Critical'], yticklabels=['Normal', 'Critical'])\nplt.ylabel('Actual Engine State')\nplt.xlabel('AI Predicted State')\nplt.title('Raw Confusion Matrix')\nplt.show()"))

    # 2. Normalized Confusion Matrix
    nb.cells.append(new_markdown_cell("### 2. Normalized Confusion Matrix (Percentages)\n**(Use: Contextual Failure Rate)** The same grid, but displayed as percentages so you can see what *percent* of total failures were successfully caught."))
    nb.cells.append(new_code_cell("cm_norm = confusion_matrix(y_test, y_pred, normalize='true')\nplt.figure(figsize=(6,5))\nsns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Normal', 'Critical'], yticklabels=['Normal', 'Critical'])\nplt.ylabel('Actual Engine State')\nplt.xlabel('AI Predicted State')\nplt.title('Normalized Confusion Matrix')\nplt.show()"))

    # 3. ROC Curve
    nb.cells.append(new_markdown_cell("### 3. ROC Curve\n**(Use: Scientific Proof)** Proves the AI is not just randomly guessing. The more the blue line pulls toward the top-left corner, the smarter the AI is."))
    nb.cells.append(new_code_cell("fpr, tpr, _ = roc_curve(y_test, y_prob)\nroc_auc = auc(fpr, tpr)\nplt.figure(figsize=(8,6))\nplt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')\nplt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\nplt.xlabel('False Alarm Rate')\nplt.ylabel('True Detection Rate')\nplt.title('Receiver Operating Characteristic (ROC)')\nplt.legend(loc='lower right')\nplt.grid(True)\nplt.show()"))

    # 4. Precision-Recall Curve
    nb.cells.append(new_markdown_cell("### 4. Precision-Recall Curve\n**(Use: Alarm Trust)** Proves that when the alarm rings, it is a real failure (Precision), and proves it catches almost all failures (Recall)."))
    nb.cells.append(new_code_cell("precision, recall, _ = precision_recall_curve(y_test, y_prob)\nplt.figure(figsize=(8,6))\nplt.plot(recall, precision, color='purple', lw=2)\nplt.xlabel('Recall (Detection Rate)')\nplt.ylabel('Precision (Trustworthiness of Alarm)')\nplt.title('Precision-Recall Curve')\nplt.grid(True)\nplt.show()"))

    # 5. Class Probability Density
    nb.cells.append(new_markdown_cell("### 5. Class Probability Distribution\n**(Use: Confidence Check)** Shows how confident the AI is. Does it sit on the fence at 50%, or is it highly confident near 0% (Normal) and 100% (Critical)?"))
    nb.cells.append(new_code_cell("plt.figure(figsize=(10,6))\nsns.kdeplot(y_prob[y_test==0], label='Actual Normal', shade=True, color='blue')\nsns.kdeplot(y_prob[y_test==1], label='Actual Critical', shade=True, color='red')\nplt.xlabel('AI Predicted Probability of Critical Failure')\nplt.title('Probability Density Distribution')\nplt.legend()\nplt.grid(True)\nplt.show()"))

    # 6. Calibration Curve
    nb.cells.append(new_markdown_cell("### 6. Calibration Curve (Reliability)\n**(Use: Trusting the AI's Output)** If the AI says '80% chance of failure', is it actually right 80% of the time? This graph proves it."))
    nb.cells.append(new_code_cell("prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)\nplt.figure(figsize=(8,6))\nplt.plot(prob_pred, prob_true, marker='o', color='green', label=model_name)\nplt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')\nplt.xlabel('Mean Predicted Probability')\nplt.ylabel('Fraction of Positives')\nplt.title('Calibration Curve')\nplt.legend()\nplt.grid(True)\nplt.show()"))

    # 7 & 8. Cumulative Gains and Lift Curve
    nb.cells.append(new_markdown_cell("### 7 & 8. Cumulative Gains and Lift Curve\n**(Use: Business Value Proof)** Shows exactly how many times faster you catch a failure using this AI versus randomly checking engines by hand."))
    nb.cells.append(new_code_cell("try:\n    import scikitplot as skplt\n    # Skplt requires a 2D array of probabilities [P(class_0), P(class_1)]\n    if len(y_prob.shape) == 1:\n        y_prob_2d = np.vstack([1-y_prob, y_prob]).T\n    else:\n        y_prob_2d = y_prob\n    fig, ax = plt.subplots(1, 2, figsize=(15, 6))\n    skplt.metrics.plot_cumulative_gain(y_test, y_prob_2d, ax=ax[0])\n    skplt.metrics.plot_lift_curve(y_test, y_prob_2d, ax=ax[1])\n    plt.show()\nexcept ImportError:\n    print('Please install scikit-plot to view Cumulative Gains and Lift Curves (pip install scikit-plot)')"))

    # 9. Feature Importance
    nb.cells.append(new_markdown_cell("### 9. Feature Importance / Coefficients\n**(Use: Maintenance Focus)** Shows exactly which sensors trigger the fire alarm the most for this specific AI model. *(Note: Not all mathematical models support this graph).*"))
    nb.cells.append(new_code_cell("try:\n    if hasattr(model, 'feature_importances_'):\n        imp = model.feature_importances_\n    elif hasattr(model, 'coef_'):\n        imp = np.abs(model.coef_[0])\n    else:\n        raise ValueError('No feature importances found for this model.')\n    \n    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': imp}).sort_values('Importance', ascending=False)\n    plt.figure(figsize=(12,6))\n    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='Reds_r')\n    plt.title(f'{model_name} Alarm Triggers (Feature Importance)')\n    plt.show()\nexcept Exception as e:\n    print(f'Cannot plot feature importance: {e}')"))
    
    # Save notebook
    class_output_dir = os.path.join("..", "classification_project", "individual_models")
    os.makedirs(class_output_dir, exist_ok=True)
    out_path = os.path.join(class_output_dir, f"{model_name}_alarm_analysis.ipynb")
    with open(out_path, 'w') as f:
        nbformat.write(nb, f)

print(f"Generated {len(classification_models)} classification notebooks.")
