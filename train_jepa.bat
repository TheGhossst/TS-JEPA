@echo off
echo ========================================
echo Starting JEPA Seed 0
echo ========================================

python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 0

IF ERRORLEVEL 1 (
    echo.
    echo Seed 0 FAILED. Seed 1 will NOT start.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Seed 0 completed successfully.
echo Starting JEPA Seed 1
echo ========================================

python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 1

IF ERRORLEVEL 1 (
    echo.
    echo Seed 1 FAILED.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ALL DONE - Seeds 0 and 1 completed!
echo ========================================
pause