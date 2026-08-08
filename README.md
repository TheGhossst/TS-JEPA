# TS-JEPA (paper-faithful baseline)

Reproduction of Time-Series JEPA per `docs/TS-JEPA_Paper-Faithful_Final.md`.

Implementation choices that are not paper-specified are listed in `docs/IMPLEMENTATION_CHOICES.md`.

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

## Pipeline

```bash
python scripts/generate_trajectories.py
python scripts/train_jepa.py
python scripts/train_actor.py
python scripts/eval_runtime.py
```

GE-JEPA is intentionally not included until this baseline is validated.
