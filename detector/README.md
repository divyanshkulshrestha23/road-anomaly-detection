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
- `rdd_prepare.py`: converts RDD2022 (VOC XML or YOLO layout) to our YOLO format and classes
- `colab_train_pothole.ipynb`: Colab notebook for Module 1 training, evaluation and TFLite export
- `train.py`: local training entry point (stub; use the Colab notebook for now)
- `export_tflite.py`: export trained weights to TFLite (stub)

## Training on Colab

Open: https://colab.research.google.com/github/divyanshkulshrestha23/road-anomaly-detection/blob/main/detector/colab_train_pothole.ipynb

Set the runtime to T4 GPU and run all cells. You need a Kaggle API token (`kaggle.json`).
Outputs go to `MyDrive/road-anomaly/` on Google Drive.

## Status

Stage 1 (pothole classes) ready to train. Sign classes will be added once a sign dataset is chosen.
