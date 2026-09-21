@echo off
echo ===================================================
echo   Starting Engine Telemetry Prediction Dashboard...
echo ===================================================
echo.

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist venv311\Scripts\activate.bat (
    call venv311\Scripts\activate.bat
)

jupyter notebook notebooks\Client_Prediction_Dashboard.ipynb
pause
