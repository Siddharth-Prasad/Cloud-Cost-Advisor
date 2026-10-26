@echo off
REM One-click setup for Windows 11 (run from the project folder)
python --version || (echo Install Python 3.11+ and tick "Add python.exe to PATH" & exit /b 1)
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python src\train_model.py
echo.
echo Setup complete. Start the app with run.bat
