# Decision log

Last updated: 2026-11-18

| # | Area | Decision | Status |
|---|---|---|---|
| 1 | Target phone | Mid-range Android, 2022-23 chipset. Record the exact model in every benchmark. | Decided |
| 2 | Vision models | One shared detector for potholes and signs, plus a vehicle detector. YOLO-style, exported to TFLite. | Decided |
| 3 | Crash rule | Crash = tilt above 50 degrees AND sudden speed drop AND stillness after impact. Impact alone never triggers an alert. | Decided |
| 4 | Confirmation | Confirmation window with cancel countdown before any SMS is sent. | Decided |
| 5 | Sign removal | Sign label is dropped once GPS shows the rider has passed the sign's stored location. | Decided |
| 6 | Distance ground truth | Measured road markers, with GPS speed as reference. | Decided |
| 7 | Provisional targets | Pothole mAP@0.5 >= 0.5. Crash recall >= 0.8 with <= 1 false alarm per hour of riding. >= 15 FPS on target phone. Revise after first baseline. | Provisional |
| 8 | Demo scope (mid-term) | Pothole module running on phone, plus sign-detector results. | Decided |
| 9 | Deadline | Mid-term review target 29-30 Nov 2026 (to be confirmed). | Open |
| 10 | Licences | Check Ultralytics (AGPL-3.0, to be verified), RDD2022, and Mapillary sign data before distribution. | Open |
| 11 | Team split | By module (see README). | Decided |
| 12 | Ownership | Module owners still to be assigned. | Open |
