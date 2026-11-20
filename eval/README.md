# Evaluation

Scripts and result tables for each module, measured against the provisional targets in `docs/decisions.md`.

- Pothole: mAP@0.5, precision, recall, FPS on target phone
- Signs: mAP, clear vs occluded accuracy
- Distance: mean absolute error against measured road markers
- Accident: crash precision, recall, false alarms per hour of riding
- Whole system: end-to-end latency, battery per hour

Results go in `results/` as CSV, one file per run.

## Distance calibration (Module 3)

1. Mount the phone on the bike as it will be used. Measure the camera lens height above the road.
2. Park a car straight ahead. Mark 5, 7.5, 10, 15, 20 and 30 m from the camera with a tape.
3. At each mark, press **Snap** in the app (not the phone's camera app: Snap saves the exact
   frame the app analyses). Repeat with a motorcycle. Measure both vehicles' real widths.
4. Copy `Android/data/dtu.rider/files/snapshots/` off the phone and rename each file with class
   and distance, e.g. `car_10m.jpg`, `motorcycle_7.5m.jpg`.
5. Run:
   ```
   python eval/distance_calibrate.py --photos snapshots/ --height 1.05 \
       --width car=1.70 motorcycle=0.78 --weights yolo11n.pt --out distance_calib.json
   ```
   It prints the per-photo error (leave-one-out) and writes `distance_calib.json`; copy it to
   `app/rider/src/main/assets/`. The CSV next to it is the distance-error table for the report.
