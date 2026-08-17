@echo off
REM Working salvage recipe (not paper). Paper protocol:
REM   python scripts/pipeline/run_full_baseline.py --config configs/ts_jepa_dp_fixed.yaml --device cuda
REM Default is seed 0 only; --all-seeds is the 5-seed working protocol.
REM --skip-generate assumes data_working/ already exists.

echo Starting working 5-seed pipeline (JEPA + gates + actor)...
python scripts/pipeline/run_working.py --config configs/ts_jepa_working.yaml --device cuda --all-seeds --skip-generate

echo All seeds completed.
pause
