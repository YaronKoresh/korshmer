@echo off
cd /d %~dp0..
call .\.venv\Scripts\activate.bat
python -m korshmer %*
call .\.venv\Scripts\deactivate.bat
pause
exit /b 0