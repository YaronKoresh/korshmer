@echo off
cd %~dp0..
call .\.venv\Scripts\activate.bat
node scripts/print_autobot_regression_fixture.cjs
call .\.venv\Scripts\deactivate.bat
pause
exit /B 0