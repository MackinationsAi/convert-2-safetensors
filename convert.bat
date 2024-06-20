@echo off
REM Get the directory of the currently executed script
set SCRIPT_DIR=%~dp0

REM Navigate to the script directory
cd /d "%SCRIPT_DIR%"

REM Activate the virtual environment
CALL venv\Scripts\activate

REM Run the conversion script
python convert_2_safetensors.py

REM Deactivate the virtual environment
CALL deactivate

exit