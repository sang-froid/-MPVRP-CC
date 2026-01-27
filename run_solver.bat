@echo off
setlocal

if "%1"=="" (
    echo Usage: run_solver.bat instance_file [time_limit]
    echo Example: run_solver.bat instances\small\MPVRP_S_001_s9_d1_p2.dat 300
    exit /b 1
)

set INSTANCE=%1
set TIME_LIMIT=%2
if "%TIME_LIMIT%"=="" set TIME_LIMIT=600

for %%F in ("%INSTANCE%") do set BASENAME=%%~nF

echo %INSTANCE% | findstr /i "small" >nul
if %ERRORLEVEL%==0 (
    set OUTPUT_DIR=solutions\small
    goto :found
)

echo %INSTANCE% | findstr /i "medium" >nul
if %ERRORLEVEL%==0 (
    set OUTPUT_DIR=solutions\medium
    goto :found
)

echo %INSTANCE% | findstr /i "large" >nul
if %ERRORLEVEL%==0 (
    set OUTPUT_DIR=solutions\large
    goto :found
)

set OUTPUT_DIR=solutions
:found

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
set OUTPUT=%OUTPUT_DIR%\Sol_%BASENAME%.dat

python solver.py "%INSTANCE%" "%OUTPUT%" %TIME_LIMIT%

endlocal