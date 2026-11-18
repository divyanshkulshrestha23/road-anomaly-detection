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
- `kaggle_train_pothole.ipynb`: Kaggle notebook for Module 1, potholes only (first baseline)
- `signs_prepare.py`: converts the sign dataset to our classes, builds 25%/50% occluded test sets, writes the combined potholes+signs config
- `signs_paste.py`: pastes sign crops into road scenes at realistic sizes (copy-paste augmentation) and builds realistic-size clear/occluded test sets
- `prepare_shared.sh`: runs all three data steps for the shared detector
- `kaggle_train_shared.ipynb`: Kaggle notebook for the shared potholes + signs detector, with clear vs occluded evaluation
- `configs/signs_map.yaml`: source sign class -> our sign group
- `colab_train_pothole.ipynb`: Colab notebook (backup) for Module 1 training, evaluation and TFLite export
- `train.py`: local training entry point (stub; use the Colab notebook for now)
- `export_tflite.py`: export trained weights to TFLite (stub)

## Training on Kaggle (primary)

Notebook: `detector/kaggle_train_pothole.ipynb`. Import it on Kaggle (File → Import Notebook), attach the `rdd-2022` dataset, set GPU and Internet on, then Save & Run All.
Outputs (weights, TFLite, summary) are in the notebook version's Output tab.

## Training on Colab (backup)

Open: https://colab.research.google.com/github/divyanshkulshrestha23/road-anomaly-detection/blob/main/detector/colab_train_pothole.ipynb

Set the runtime to T4 GPU and run all cells. You need a Kaggle API token (store it as a Colab secret named `KAGGLE_API_TOKEN`).
Outputs go to `MyDrive/road-anomaly/` on Google Drive.

## Status

Stage 1 (pothole classes) ready to train. Sign classes will be added once a sign dataset is chosen.
