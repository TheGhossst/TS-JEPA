@echo off

echo Starting JEPA seed 0...
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 0 --epoch 60

echo Seed 0 finished. Starting JEPA seed 1...
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 1 --epoch 60

echo Seed 1 finished. Starting JEPA seed 2...
python scripts/pipeline/train_jepa.py --config configs/ts_jepa_dp_fixed.yaml --device cuda --single-seed 2 --epoch 60

echo All seeds completed.
pause