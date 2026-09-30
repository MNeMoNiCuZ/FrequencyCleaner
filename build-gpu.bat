@echo off
:: Builds FrequencyCleaner-GPU.exe (with torch/CUDA) with PyInstaller and moves it
:: (plus its _internal_gpu folder) to this folder, next to input\, output\,
:: presets\ and history.json.
setlocal
cd /d "%~dp0"

set "PY=venv\Scripts\python.exe"
set "NAME=FrequencyCleaner-GPU"
set "CONTENTS=_internal_gpu"
set "WORK=build_gpu_tmp"

if not exist "%PY%" (
    echo venv not found - run venv_create.bat first.
    goto fail
)

"%PY%" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo CUDA-enabled torch not found in the venv.
    goto fail
)

"%PY%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller into the venv ...
    "%PY%" -m pip install pyinstaller
    if errorlevel 1 goto fail
)

echo.
echo Building %NAME% ...
if exist "%WORK%" rmdir /s /q "%WORK%"
"%PY%" -m PyInstaller --noconfirm --clean --windowed --onedir ^
    --name "%NAME%" ^
    --icon "%~dp0src\assets\frequency_cleaner.ico" ^
    --contents-directory "%CONTENTS%" ^
    --distpath "%WORK%\dist" --workpath "%WORK%\build" --specpath "%WORK%" ^
    --add-data "%~dp0src\icons;icons" ^
    --add-data "%~dp0src\assets\frequency_cleaner.png;assets" ^
    --hidden-import torch ^
    --exclude-module torchvision --exclude-module torchaudio ^
    src\main.py
if errorlevel 1 goto fail

echo.
echo Moving the build to %~dp0 ...
if exist "%NAME%.exe" del /q "%NAME%.exe"
if exist "%CONTENTS%" rmdir /s /q "%CONTENTS%"
move "%WORK%\dist\%NAME%\%NAME%.exe" . >nul
if errorlevel 1 goto fail
move "%WORK%\dist\%NAME%\%CONTENTS%" . >nul
if errorlevel 1 goto fail
rmdir /s /q "%WORK%"

:: PyQt6 ships an old MSVC runtime (14.26) in Qt6\bin, which the frozen app puts
:: on the DLL path; torch's c10.dll fails to initialise against it. Remove it so
:: the newer runtime in %CONTENTS% is used.
for %%D in (MSVCP140 MSVCP140_1 MSVCP140_2 VCRUNTIME140 VCRUNTIME140_1) do (
    if exist "%CONTENTS%\PyQt6\Qt6\bin\%%D.dll" del /q "%CONTENTS%\PyQt6\Qt6\bin\%%D.dll"
    if not exist "%CONTENTS%\%%D.dll" if exist "%SystemRoot%\System32\%%D.dll" copy /y "%SystemRoot%\System32\%%D.dll" "%CONTENTS%\" >nul
)

echo.
echo Done: %~dp0%NAME%.exe
pause
exit /b 0

:fail
echo.
echo Build FAILED.
pause
exit /b 1
