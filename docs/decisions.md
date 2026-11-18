# Decision log

Last updated: 2026-11-18 (rev 2)

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
| 9 | Deadline | Mid-term review 29-30 Nov 2026. | Decided |
| 10 | Licences | RDD2022: CC BY-SA 4.0, attribution required (confirmed). Still to check: Ultralytics (AGPL-3.0, to be verified) and the sign dataset. | Partly done |
| 11 | Team split | By module (see README). | Decided |
| 12 | Ownership | Module owners still to be assigned (on hold). | Open |
| 13 | Training compute | Kaggle Notebooks (free GPU; dataset attached directly, no download). Colab kept as backup; free Colab compute units were too low. | Decided |
| 14 | Sign classes | regulatory, warning, hospital, school, speed_limit. | Decided |
| 15 | Pothole training data | RDD2022 via Kaggle; 4 classes (D00, D10, D20, D40); China drone subset excluded (aerial view); other classes dropped; background images capped at 20%; split 80/10/10 per country; India-only val/test reported separately. | Decided |
| 17 | Android stack | Kotlin, CameraX, TensorFlow Lite (FP16 model, GPU delegate when available). Gradle root `app/`, module `rider/`, package `dtu.rider`. | Decided |
| 18 | Sign dataset | Candidate: Roboflow "indian traffic signs detection 2" (CC BY 4.0, 37 classes). Backup: IRTSD (needs IEEE DataPort subscription). To confirm after export. | Open |
| 16 | Shared detector order | Stage 1 trains pothole classes only (RDD2022 has no signs). Signs are added once a sign dataset is chosen, then the model is retrained on all classes. | Decided |
