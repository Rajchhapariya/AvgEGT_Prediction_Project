@echo off
echo ===================================================
echo   Starting the AI Engine Prediction Dashboard...
echo ===================================================
echo.
echo Loading the AI Brains (this may take a few seconds)...

:: Activate the virtual environment
call venv311\Scripts\activate.bat

:: Launch the Jupyter Notebook directly into the dashboard
jupyter notebook notebooks\Client_Prediction_Dashboard.ipynb

pause
