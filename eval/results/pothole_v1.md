# Module 1 baseline: pothole + crack detector (run `rdd_pothole_v1`)

- Model: YOLO11n (Ultralytics 8.4.215), 640 px, batch 32, single Kaggle T4
- Data: RDD2022 Kaggle copy, dataset's own split, China drone subset excluded,
  "other corruption" dropped, train background capped at 20% (19,237 train / 5,409 val / 5,381 test images)
- Training: 138 epochs (~4 min each, ~9.5 h), early stopping (patience 15); best epoch 123
- Checkpoint `best.pt`: 5.5 MB, saved 2026-11-19 08:09 IST

## Results (from the run log)

| Set | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| All countries, val | 0.547 | 0.280 | 0.622 | 0.510 |
| **All countries, test** | **0.555** | **0.281** | **0.630** | **0.517** |
| India only, val (1,172 images) | 0.407 | 0.183 | 0.436 | 0.450 |
| **India only, test** | **0.364** | **0.161** | **0.432** | **0.326** |

Per-class mAP@0.5:0.95, test:

| Class | All countries | India |
|---|---|---|
| pothole | 0.208 | 0.176 |
| longitudinal_crack | 0.313 | 0.157 |
| transverse_crack | 0.246 | 0.005 |
| alligator_crack | 0.359 | 0.304 |

## Reading

- Provisional target (pothole mAP@0.5 >= 0.5) is met on the all-countries test set (0.555),
  but not on Indian roads (0.364). The India gap is the main thing to improve.
- India transverse cracks are almost never detected (0.005). Instance counts per class on the
  India test set were not logged; check them before drawing conclusions (likely very few).
- Validation mAP@0.5 plateaued at ~0.54 from about epoch 90; more epochs alone will not help much.
- The INT8 TFLite export ran out of memory after training (static calibration); FP16 export is
  done separately with `detector/export_tflite.py`.

## Ideas for the next run

- Fine-tune on India only (start from this `best.pt`) or oversample India in training.
- Larger input (e.g. 800 px) for small potholes, if phone FPS allows.
- Report per-class instance counts with each result.
