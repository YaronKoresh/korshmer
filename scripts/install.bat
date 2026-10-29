@echo off
cd /d %~dp0

set "PYTHON_DIR=%LOCALAPPDATA%\Programs\Python\Python311"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"

for /f "tokens=*" %%i in ('dir /b /s "C:\Program Files (x86)\Windows Kits\10\bin\*signtool.exe" 2^>nul') do (
    set "SIGNTOOL_PATH=%%i"
)

if "%1"=="install_win_sdk" goto install_win_sdk_prompt
if "%1"=="install_ffmpeg" goto install_ffmpeg_prompt
if "%1"=="install_clang" goto install_clang_prompt

echo Checking for prerequisites...

if exist "%ProgramFiles%\Python311\python.exe" (
    echo [WARN] An all-users Python 3.11 exists in Program Files.
    echo [WARN] Uninstall it from Settings - Apps, otherwise "python" on PATH may point to it.
)

if not exist "%PYTHON_EXE%" (
    echo [INFO] Python not found at "%PYTHON_EXE%".
    goto install_python_prompt
)
:after_python
echo  - Python: Found at "%PYTHON_EXE%"

if not defined SIGNTOOL_PATH (
    echo [INFO] Signtool not found. Administrator privileges are required to install it.
    goto install_win_sdk_prompt
)
echo  - Signtool: Found at: "%SIGNTOOL_PATH%"

where ffmpeg.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] FFmpeg not found. Administrator privileges are required to install it.
    goto install_ffmpeg_prompt
)
echo  - FFmpeg: Found.

where clang.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Clang C++ Compiler not found. It will be installed automatically.
    goto install_clang_prompt
)
echo  - C++ Compiler (Clang): Found.

goto install_package

:install_python_prompt
echo [INFO] Now installing Python 3.11.5 for the current user...
set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe"
set "INSTALLER_PATH=%TEMP%\python_installer.exe"
echo Downloading Python 3.11.5 installer...
powershell -NoProfile -Command "(New-Object System.Net.WebClient).DownloadFile('%PYTHON_INSTALLER_URL%', '%INSTALLER_PATH%')"
if %errorlevel% neq 0 ( goto end_error )
start /wait "" "%INSTALLER_PATH%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 TargetDir="%PYTHON_DIR%"
del "%INSTALLER_PATH%"
powershell -NoProfile -Command "$d='%PYTHON_DIR%'; $s=$d+'\Scripts'; $p=[Environment]::GetEnvironmentVariable('Path','User'); $parts=@($p -split ';' | Where-Object { $_ -and $_ -ne $d -and $_ -ne $s }); [Environment]::SetEnvironmentVariable('Path',((@($d,$s)+$parts) -join ';'),'User')"
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python installation did not create "%PYTHON_EXE%".
    echo [ERROR] If an all-users Python 3.11.5 is installed, uninstall it first and re-run.
    goto end_error
)
echo [SUCCESS] Python installed.
goto after_python

:install_win_sdk_prompt
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process '%~f0' -Verb RunAs -ArgumentList 'install_win_sdk'"
    exit /b
)
echo [INFO] Now running as Administrator to install Windows SDK...
echo [INFO] Attempting to install using Winget (Windows Package Manager)...
winget install --id=Microsoft.WindowsSDK --silent --accept-source-agreements --accept-package-agreements
if %errorlevel% equ 0 (
    echo [SUCCESS] Windows SDK has been installed via Winget. Please re-run this script.
    goto end_success
)
echo [WARN] Winget installation failed. Attempting manual download...
set "WSDK_URL=https://go.microsoft.com/fwlink/?linkid=2376217"
set "WSDK_INSTALLER_PATH=%TEMP%\winsdksetup.exe"
echo Downloading latest Windows SDK installer...
powershell -NoProfile -Command "(New-Object System.Net.WebClient).DownloadFile('%WSDK_URL%', '%WSDK_INSTALLER_PATH%')"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download Windows SDK.
    goto end_error
)
start /wait "" "%WSDK_INSTALLER_PATH%" /features OptionId.SigningTools /quiet /norestart
del "%WSDK_INSTALLER_PATH%"
echo [SUCCESS] Windows SDK has been installed manually. Please re-run this script from a new command prompt.
goto end_success

:install_ffmpeg_prompt
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process '%~f0' -Verb RunAs -ArgumentList 'install_ffmpeg'"
    exit /b
)
echo [INFO] Now running as Administrator to install FFmpeg...
echo [INFO] Attempting to install using Winget (Windows Package Manager)...
winget install --id=Gyan.FFmpeg.Essentials -e --accept-source-agreements --accept-package-agreements
if %errorlevel% equ 0 (
    echo [SUCCESS] FFmpeg has been installed via Winget. Please re-run this script.
    goto end_success
)
echo [WARN] Winget installation failed. Attempting manual download...
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "ZIP_PATH=%TEMP%\ffmpeg.zip"
set "EXTRACT_PATH=%TEMP%\ffmpeg_extracted"
echo Downloading latest FFmpeg build...
powershell -NoProfile -Command "(New-Object System.Net.WebClient).DownloadFile('%FFMPEG_URL%', '%ZIP_PATH%')"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download FFmpeg.
    goto end_error
)
echo Extracting FFmpeg...
powershell -NoProfile -Command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '%EXTRACT_PATH%' -Force"
md "%ProgramFiles%\ffmpeg" >nul 2>nul
for /d %%d in ("%EXTRACT_PATH%\*") do (
    set "FFMPEG_DIR=%%d"
)
robocopy "%FFMPEG_DIR%\bin" "%ProgramFiles%\ffmpeg" /E /MOVE >nul
echo Adding FFmpeg to the system PATH...
powershell -NoProfile -Command "$p=[Environment]::GetEnvironmentVariable('Path','Machine'); if($p -notlike '*\ffmpeg*'){[Environment]::SetEnvironmentVariable('Path',($p.TrimEnd(';')+';%ProgramFiles%\ffmpeg'),'Machine')}"
del "%ZIP_PATH%"
rmdir /s /q "%EXTRACT_PATH%"
echo [SUCCESS] FFmpeg has been installed manually. Please re-run this script from a new command prompt.
goto end_success

:install_clang_prompt
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -NoProfile -Command "Start-Process '%~f0' -Verb RunAs -ArgumentList 'install_clang'"
    exit /b
)
echo [INFO] Now running as Administrator to install Clang...
set "CLANG_INSTALLER_URL=https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/LLVM-18.1.8-win64.exe"
set "INSTALLER_PATH=%TEMP%\clang_installer.exe"
powershell -NoProfile -Command "(New-Object System.Net.WebClient).DownloadFile('%CLANG_INSTALLER_URL%', '%INSTALLER_PATH%')"
if %errorlevel% neq 0 ( goto end_error )
echo Starting Clang installation (this may take a few minutes)...
setx /M LLVM_INSTALL_DIR "%ProgramFiles%\LLVM" >nul
start /wait "" "%INSTALLER_PATH%" /S /D=%ProgramFiles%\LLVM
powershell -NoProfile -Command "$p=[Environment]::GetEnvironmentVariable('Path','Machine'); if($p -notlike '*\LLVM\bin*'){[Environment]::SetEnvironmentVariable('Path',($p.TrimEnd(';')+';%ProgramFiles%\LLVM\bin'),'Machine')}"
del "%INSTALLER_PATH%"
echo [SUCCESS] Clang C++ Compiler has been installed. Please re-run this script from a new command prompt.
goto end_success

:install_package
echo Installing/Updating Korshmer and all dependencies...
"%PYTHON_EXE%" -m venv ..\.venv
call ..\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e "..[dev]"
python -m poethepoet hook
call ..\.venv\Scripts\deactivate.bat
if %errorlevel% neq 0 (
    goto end_error
)

echo.
echo [SUCCESS] Korshmer has been installed successfully.
echo Python used: %PYTHON_EXE%
goto end_success

:end_error
echo.
echo [INSTALLATION FAILED] An error occurred.
echo.
pause
exit /b 1

:end_success
pause
exit /b 0