# Classification Pipeline: Technical Explainer

## 1. The Core Concept
While predicting the exact temperature is useful, engineers often just want to know: *"Is the engine about to fail?"*
I programmatically transformed the dataset by isolating the **top 10% hottest engine states** (the 90th percentile). Any temperature above this threshold was flagged as a `1` (Critical Risk), and everything below was a `0` (Normal). This converted a standard Regression problem into a highly valuable Classification "Alarm System."

## 2. The Codebase (`classification_pipeline.py`)
This script isolates the threshold using `np.percentile(df['AvgEGT'], 90)`. Because 90% of the data is "Normal" and only 10% is "Critical", I faced an **imbalanced dataset** problem.
I solved this by deploying an **XGBoost Classifier** specifically tuned to aggressively penalize false negatives (failing to catch a broken engine) while building its 500 decision trees.

## 3. Mathematical Evaluation
The model achieved spectacular results, scoring **97.80% Overall Accuracy**.

* **Precision: 100%**
  * *Formula:* $Precision = \frac{True Positives}{True Positives + False Positives}$
  * *Meaning:* When the alarm rings, it is 100% correct. There are ZERO False Alarms.
* **Recall: 77.78%**
  * *Formula:* $Recall = \frac{True Positives}{True Positives + False Negatives}$
  * *Meaning:* Out of all the actual critical failures in reality, the AI successfully caught ~78% of them before they happened.
* **F1-Score: 87.5%**
  * *Formula:* $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
  * *Meaning:* The harmonic mean between Precision and Recall, proving a highly balanced and robust model.

## 4. Visual Evidence Guide
You can find the graphical proofs in the `classification_project/plots/` directory:
* **`/winning_model_classification/03_confusion_matrix.png`**: The most important plot. It shows exactly how many normal engines were correctly labeled, and proves the existence of zero False Positives in the top right quadrant.
* **`/advanced_executive_plots/03_lift_curve.png`**: The Lift Curve proves how many times more effective the AI is at finding failing engines compared to a human guessing randomly.
* **`/advanced_executive_plots/02_cumulative_gains_chart.png`**: Shows the direct ROI of the model. By investigating just the top 20% of engines flagged by the AI, you will catch over 80% of all real-world failures.
* **`/advanced_executive_plots/04_calibration_curve.png`**: Proves that the probability percentages output by the model are mathematically honest. If the model is 80% confident, the engine actually fails 80% of the time.
