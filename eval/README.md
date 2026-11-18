# Evaluation

Scripts and result tables for each module, measured against the provisional targets in `docs/decisions.md`.

- Pothole: mAP@0.5, precision, recall, FPS on target phone
- Signs: mAP, clear vs occluded accuracy
- Distance: mean absolute error against measured road markers
- Accident: crash precision, recall, false alarms per hour of riding
- Whole system: end-to-end latency, battery per hour

Results go in `results/` as CSV, one file per run.
