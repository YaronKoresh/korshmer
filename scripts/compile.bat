@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] The Python environment was not found. Run scripts\install.bat first.
    goto failed
)
call ".venv\Scripts\activate.bat"

echo Installing the compiler...
python -m pip install -e ".[build]"
if errorlevel 1 goto failed

set "APP_VERSION="
for /f "usebackq delims=" %%v in (`python -c "import korshmer; print(korshmer.__version__)"`) do set "APP_VERSION=%%v"
if not defined APP_VERSION (
    echo [ERROR] Could not read the application version. Run scripts\install.bat first.
    goto failed
)

echo Downloading the models...
python src\korshmer\download_models.py
if errorlevel 1 goto failed

set "EXTRA_DATA="
if exist "models\" set "EXTRA_DATA=%EXTRA_DATA% --include-data-dir=models=models"
if exist "fluidsynth\" set "EXTRA_DATA=%EXTRA_DATA% --include-raw-dir=fluidsynth=fluidsynth"
if exist "assets\icon.ico" set "EXTRA_DATA=%EXTRA_DATA% --windows-icon-from-ico=assets\icon.ico"
if not exist "fluidsynth\" echo [WARNING] The fluidsynth folder is missing, so the built program will have no instrument sounds.

echo Compiling KORSHMER %APP_VERSION%. This can take a long time...
set MPLBACKEND=agg
python -m nuitka ^
--mode=standalone ^
--python-flag=-m ^
--file-version=%APP_VERSION% ^
--product-version=%APP_VERSION% ^
--assume-yes-for-downloads ^
--company-name="Yaron Koresh" ^
--copyright="Copyright (c) 2025 Yaron Koresh" ^
--deployment ^
--enable-plugin=pyside6 ^
--file-description="KORSHMER audio workstation" ^
--include-package-data=numba ^
--include-package-data=basic_pitch ^
--include-package-data=demucs ^
--include-package=basic_pitch ^
--include-package=demucs ^
--include-package=librosa ^
--include-package=onnxruntime ^
--include-package=sklearn.cluster ^
--include-package=soundfile ^
--jobs=%NUMBER_OF_PROCESSORS% ^
--lto=yes ^
--module-parameter=numba-disable-jit=yes ^
--module-parameter=torch-disable-jit=yes ^
--nofollow-import-to=*.conftest ^
--nofollow-import-to=_pytest ^
--nofollow-import-to=absl ^
--nofollow-import-to=astunparse ^
--nofollow-import-to=azure ^
--nofollow-import-to=basic_pitch.command_line_interface ^
--nofollow-import-to=basic_pitch.data ^
--nofollow-import-to=basic_pitch.train ^
--nofollow-import-to=boto ^
--nofollow-import-to=boto3 ^
--nofollow-import-to=botocore ^
--nofollow-import-to=celery ^
--nofollow-import-to=chardet ^
--nofollow-import-to=cv2 ^
--nofollow-import-to=cython ^
--nofollow-import-to=dask ^
--nofollow-import-to=demucs.augment ^
--nofollow-import-to=demucs.evaluate ^
--nofollow-import-to=demucs.grids ^
--nofollow-import-to=demucs.propagate ^
--nofollow-import-to=demucs.repitch ^
--nofollow-import-to=demucs.solver ^
--nofollow-import-to=demucs.train ^
--nofollow-import-to=docutils ^
--nofollow-import-to=flax ^
--nofollow-import-to=gast ^
--nofollow-import-to=gi ^
--nofollow-import-to=google.cloud ^
--nofollow-import-to=gradio ^
--nofollow-import-to=h5py ^
--nofollow-import-to=huggingface_hub ^
--nofollow-import-to=IPython ^
--nofollow-import-to=jax ^
--nofollow-import-to=jaxlib ^
--nofollow-import-to=keras ^
--nofollow-import-to=kombu ^
--nofollow-import-to=librosa.display ^
--nofollow-import-to=llvmlite.tests ^
--nofollow-import-to=matplotlib.tests ^
--nofollow-import-to=numba.cuda ^
--nofollow-import-to=numba.tests ^
--nofollow-import-to=onnxruntime.quantization ^
--nofollow-import-to=onnxruntime.tools ^
--nofollow-import-to=onnxruntime.transformers ^
--nofollow-import-to=pandas ^
--nofollow-import-to=pygments ^
--nofollow-import-to=PySide6.QtBluetooth ^
--nofollow-import-to=PySide6.QtCharts ^
--nofollow-import-to=PySide6.QtDataVisualization ^
--nofollow-import-to=PySide6.QtDesigner ^
--nofollow-import-to=PySide6.QtHelp ^
--nofollow-import-to=PySide6.QtMultimedia ^
--nofollow-import-to=PySide6.QtMultimediaWidgets ^
--nofollow-import-to=PySide6.QtNetwork ^
--nofollow-import-to=PySide6.QtNfc ^
--nofollow-import-to=PySide6.QtPdf ^
--nofollow-import-to=PySide6.QtPdfWidgets ^
--nofollow-import-to=PySide6.QtQml ^
--nofollow-import-to=PySide6.QtQuick ^
--nofollow-import-to=PySide6.QtQuick3D ^
--nofollow-import-to=PySide6.QtQuickWidgets ^
--nofollow-import-to=PySide6.QtRemoteObjects ^
--nofollow-import-to=PySide6.QtScxml ^
--nofollow-import-to=PySide6.QtSensors ^
--nofollow-import-to=PySide6.QtSerialPort ^
--nofollow-import-to=PySide6.QtSql ^
--nofollow-import-to=PySide6.QtStateMachine ^
--nofollow-import-to=PySide6.QtSvg ^
--nofollow-import-to=PySide6.QtSvgWidgets ^
--nofollow-import-to=PySide6.QtTest ^
--nofollow-import-to=PySide6.QtUiTools ^
--nofollow-import-to=PySide6.QtVirtualKeyboard ^
--nofollow-import-to=PySide6.QtWebChannel ^
--nofollow-import-to=PySide6.QtWebEngine ^
--nofollow-import-to=PySide6.QtWebEngineCore ^
--nofollow-import-to=PySide6.QtWebEngineWidgets ^
--nofollow-import-to=PySide6.QtWebSockets ^
--nofollow-import-to=PySide6.QtXml ^
--nofollow-import-to=pytest ^
--nofollow-import-to=regex ^
--nofollow-import-to=rsa ^
--nofollow-import-to=s3transfer ^
--nofollow-import-to=safetensors ^
--nofollow-import-to=sklearn.cluster.tests ^
--nofollow-import-to=tensorboard ^
--nofollow-import-to=tensorboard_plugin_wit ^
--nofollow-import-to=tensorflow ^
--nofollow-import-to=tensorflow_estimator ^
--nofollow-import-to=tensorflow_intel ^
--nofollow-import-to=tensorflow_probability ^
--nofollow-import-to=tkinter ^
--nofollow-import-to=tokenizers ^
--nofollow-import-to=torch._dynamo ^
--nofollow-import-to=torch._inductor ^
--nofollow-import-to=torch._lazy ^
--nofollow-import-to=torch._numpy ^
--nofollow-import-to=torch.onnx ^
--nofollow-import-to=torch.utils.benchmark ^
--nofollow-import-to=torch.utils.tensorboard ^
--nofollow-import-to=tornado ^
--nofollow-import-to=transformers ^
--nofollow-import-to=wrapt ^
--nofollow-import-to=zmq ^
--noinclude-default-mode=warning ^
--output-dir=build ^
--output-filename=korshmer.exe ^
--product-name="KORSHMER" ^
--python-flag=no_asserts ^
--python-flag=no_docstrings ^
--python-flag=no_warnings ^
--user-package-configuration-file=scripts\nuitka-package.config.yml ^
--windows-console-mode=disable ^
%EXTRA_DATA% ^
src\korshmer
if errorlevel 1 goto failed

if not exist "build\korshmer.dist\korshmer.exe" (
    echo [ERROR] The compiler finished but build\korshmer.dist\korshmer.exe was not created.
    goto failed
)

echo.
echo [SUCCESS] KORSHMER %APP_VERSION% was built in build\korshmer.dist
call ".venv\Scripts\deactivate.bat"
pause
exit /b 0

:failed
echo.
echo [BUILD FAILED] See the messages above.
pause
exit /b 1
