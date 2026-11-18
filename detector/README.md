# Detector (Python)

Training, dataset configuration and TFLite export for the shared pothole + sign detector, and the vehicle detector.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

- `configs/`: dataset and class definitions (`classes.yaml` is the single source of truth for class names)
- `train.py`: training entry point (stub until dataset is locked)
- `export_tflite.py`: export trained weights to TFLite (stub)

## Status

Skeleton only. Dataset, class list and model choice are pending the first baseline.
