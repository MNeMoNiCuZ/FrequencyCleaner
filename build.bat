@echo off
:: Builds FrequencyCleaner.exe with PyInstaller and moves it (plus its _internal
:: folder) to this folder, next to input\, output\, presets\ and history.json.
setlocal
cd /d "%~dp0"

set "PY=venv\Scripts\python.exe"
set "NAME=FrequencyCleaner"
set "WORK=build_tmp"

if not exist "%PY%" (
    echo venv not found - run venv_create.bat first.
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
    --distpath "%WORK%\dist" --workpath "%WORK%\build" --specpath "%WORK%" ^
    --add-data "%~dp0src\icons;icons" ^
    --add-data "%~dp0src\assets\frequency_cleaner.png;assets" ^
    --exclude-module torch --exclude-module torchvision --exclude-module torchaudio ^
    src\main.py
if errorlevel 1 goto fail

echo.
echo Moving the build to %~dp0 ...
if exist "%NAME%.exe" del /q "%NAME%.exe"
if exist "_internal" rmdir /s /q "_internal"
move "%WORK%\dist\%NAME%\%NAME%.exe" . >nul
if errorlevel 1 goto fail
move "%WORK%\dist\%NAME%\_internal" . >nul
if errorlevel 1 goto fail
rmdir /s /q "%WORK%"

echo.
echo Done: %~dp0%NAME%.exe
pause
exit /b 0

:fail
echo.
echo Build FAILED.
pause
exit /b 1
