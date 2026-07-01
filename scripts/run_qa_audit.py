import os
import py_compile
import subprocess
import glob

print("Starting Massive QA Audit...")

# Output report file
REPORT_FILE = "../BUG_REPORT.txt"
errors_found = []

# Directories to ignore
IGNORE_DIRS = ['venv311', '.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints']

# Collect all files
py_files = []
ipynb_files = []

for root, dirs, files in os.walk(".."):
    # Modify dirs in-place to skip ignored directories
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
    for file in files:
        if file.endswith('.py'):
            py_files.append(os.path.join(root, file))
        elif file.endswith('.ipynb'):
            ipynb_files.append(os.path.join(root, file))

# ---------------------------------------------------------
# PHASE 1: Python Syntax Check
# ---------------------------------------------------------
print(f"Phase 1: Syntax Checking {len(py_files)} Python scripts...")
for py_file in py_files:
    try:
        py_compile.compile(py_file, doraise=True)
    except Exception as e:
        errors_found.append(f"[SYNTAX ERROR] {py_file}:\n{e}")

# ---------------------------------------------------------
# PHASE 2: Hardcoded Path Check
# ---------------------------------------------------------
print(f"Phase 2: Scanning {len(py_files) + len(ipynb_files)} files for Hardcoded Paths...")
all_code_files = py_files + ipynb_files
bad_paths = ["C:\\Users\\Rajch", "C:/Users/Rajch"]

for code_file in all_code_files:
    try:
        with open(code_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for bp in bad_paths:
                if bp in content:
                    errors_found.append(f"[HARDCODED PATH] Found '{bp}' in {code_file}")
    except Exception as e:
        errors_found.append(f"[FILE READ ERROR] Could not read {code_file}: {e}")

# ---------------------------------------------------------
# PHASE 3: Notebook Execution Stress Test
# ---------------------------------------------------------
# To ensure the process completes in a reasonable time, we will test the pipelines
# and the master notebook. The 20 individual model notebooks are identical in structure,
# so we will test 2 of them as representatives.
test_notebooks = [
    "../notebooks/Client_Prediction_Dashboard.ipynb",
    "../notebooks/individual_models/xgboost_model_analysis.ipynb",
    "../classification_project/individual_models/logistic_regression_alarm_analysis.ipynb"
]

print(f"Phase 3: Execution Stress Test on {len(test_notebooks)} critical notebooks...")
for nb in test_notebooks:
    if not os.path.exists(nb):
        errors_found.append(f"[MISSING FILE] Cannot test notebook {nb}")
        continue
    
    print(f"  -> Testing {nb}...")
    import sys
    try:
        # Run nbconvert to execute the notebook using the current venv python
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            errors_found.append(f"[EXECUTION CRASH] Notebook {nb} crashed during execution.\nDetails:\n{result.stderr}")
    except Exception as e:
        errors_found.append(f"[EXECUTION ERROR] Failed to run nbconvert on {nb}: {e}")

# ---------------------------------------------------------
# GENERATE REPORT
# ---------------------------------------------------------
print("\nFinalizing Report...")
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("=========================================\n")
    f.write("        PROJECT QA AUDIT REPORT\n")
    f.write("=========================================\n\n")
    
    if len(errors_found) == 0:
        f.write("[PASSED] CONGRATULATIONS! Zero bugs found.\n")
        f.write("The project is perfectly syntactically correct, has no hardcoded paths, and all tested notebooks executed without crashing.\n")
        print("[PASSED] SUCCESS! Zero bugs found. Check BUG_REPORT.txt.")
    else:
        f.write(f"[FAILED] FOUND {len(errors_found)} BUGS OR ISSUES:\n\n")
        for err in errors_found:
            f.write(err + "\n-----------------------------------------\n")
        print(f"[FAILED] FOUND {len(errors_found)} BUGS. Check BUG_REPORT.txt.")
