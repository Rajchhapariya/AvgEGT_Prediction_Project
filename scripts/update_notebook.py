import nbformat

path = "c:/Users/Rajch/Desktop/AvgEGT_Prediction_Project/notebooks/Client_Prediction_Dashboard.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        if "from xgboost import XGBRegressor, XGBClassifier" in source:
            source = source.replace("from xgboost import XGBRegressor, XGBClassifier", "import joblib")
            old_train = "# 2. Train Models (XGBoost is fast enough to do this live for the client)\\nreg_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)\\nreg_model.fit(X_scaled, y_reg)\\n\\ncls_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)\\ncls_model.fit(X_scaled, y_cls)"
            # Wait, since the source string inside the notebook object is actual raw text (no \n escaped as string), let's use standard replacement:
            source = source.replace("# 2. Train Models (XGBoost is fast enough to do this live for the client)", "# 2. Load Pre-trained Models")
            source = source.replace("reg_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)\nreg_model.fit(X_scaled, y_reg)", "reg_model = joblib.load(os.path.join('..', 'final_model.pkl'))")
            source = source.replace("cls_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)\ncls_model.fit(X_scaled, y_cls)", "cls_model = joblib.load(os.path.join('..', 'classification_project', 'classification_model.pkl'))")
            cell.source = source

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Updated Notebook.")
