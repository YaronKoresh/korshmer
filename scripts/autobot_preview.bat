@echo off
cd %~dp0..
call .\.venv\Scripts\activate.bat
python -m poethepoet autobot-preview
call .\.venv\Scripts\deactivate.bat
pause
exit /B 0