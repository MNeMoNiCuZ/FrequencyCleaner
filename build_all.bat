@echo off
:: Builds FrequencyCleaner-GPU.exe (build-gpu.bat), then FrequencyCleaner.exe (build.bat).
setlocal
cd /d "%~dp0"

set "GPU=OK"
set "CPU=OK"

:: stdin from nul skips the pause at the end of each build script.
call "%~dp0build_gpu.bat" <nul
if errorlevel 1 set "GPU=FAILED"

call "%~dp0build.bat" <nul
if errorlevel 1 set "CPU=FAILED"

echo.
echo ==============================
echo  FrequencyCleaner-GPU.exe : %GPU%
echo  FrequencyCleaner.exe     : %CPU%
echo ==============================
pause
if "%GPU%%CPU%"=="OKOK" exit /b 0
exit /b 1
