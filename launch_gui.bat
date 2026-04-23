@echo off
title Blood Cell Classifier GUI
echo ========================================
echo    Blood Cell Classifier GUI
echo ========================================
echo.

call venv_bcc\Scripts\activate

echo Loading model and starting GUI...
python launch_gui.py

pause