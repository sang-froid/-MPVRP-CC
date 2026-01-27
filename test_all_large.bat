@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo MPVRP-CC OR-Tools Solver - SMALL Instances
echo ============================================
echo.

set INSTANCES_DIR=instances\small
set TIME_LIMIT=300
set TOTAL=0
set SUCCESS=0

if not exist "solutions\small" mkdir "solutions\small"

for %%F in (%INSTANCES_DIR%\*.dat) do (
    set /a TOTAL+=1
    echo.
    echo [!TOTAL!] Processing: %%~nxF
    echo ------------------------------------------------
    
    call run_solver.bat "%%F" %TIME_LIMIT%
    
    if !ERRORLEVEL! EQU 0 (
        set /a SUCCESS+=1
    )
)

echo.
echo ================================================
echo SUMMARY
echo ================================================
echo Total: !TOTAL!
echo Success: !SUCCESS!
echo ================================================

endlocal
pause