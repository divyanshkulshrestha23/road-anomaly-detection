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
| 17 | Android stack | Kotlin, CameraX, LiteRT 1.4.2 (TensorFlow Lite Interpreter API; FP32 model, GPU delegate with CPU fallback). Gradle root `app/`, module `rider/`, package `dtu.rider`. | Decided |
| 18 | Sign dataset | Candidate: Roboflow "Indian Traffic SignBoards" (major-project-166na, MIT, 75 classes); the earlier 37-class candidate was removed from Roboflow. Backup: IRTSD (needs IEEE DataPort subscription). Mapping uses keyword rules so any Indian sign dataset works. To confirm after export. | Open |
| 19 | Occlusion evaluation | Sign test set copied with 25% and 50% of each sign covered by a patch from another road image, entering from a random side. Train also gets occluded copies of 30% of sign images. Report mAP clear vs 25% vs 50%. | Decided |
| 21 | Sign scale | Sign dataset is mostly close-ups (median box 62% of image, 3% under 1%). Copy-paste augmentation: sign crops scaled to 2.5-12% of frame width and pasted roadside (70% left) into RDD scenes of the same split, keeping RDD damage labels. Train gets 2 composites per sign image; val/test get 1, plus 25%/50% occluded test copies with identical placement. Report close-up and realistic-size sign results. | Decided |
| 22 | Rare sign classes | hospital (46 train / 8 test boxes) and school (40 / 20) are rare. Train gets 6 pasted composites per hospital/school image instead of 2. Test counts stay small, so per-class hospital/school scores are noisy; report counts alongside them. | Decided |
| 20 | Known limitation | Merged training: RDD images contain unlabelled signs and sign images contain unlabelled road damage, which can depress recall. Note in the report; revisit if sign recall is low. | Noted |
| 23 | Vehicle detector | COCO-pretrained YOLO11n (no training), FP32 LiteRT, every 2nd frame; classes car, motorcycle, bus, truck, bicycle. Auto-rickshaws are not a COCO class and may be missed or mislabelled. | Decided |
| 24 | Distance method | Width-based (assumed vehicle widths) fused with ground-plane (camera height + horizon) when they agree within 2x; calibrated from Snap frames at taped distances; alpha-beta tracker (0.4/0.1) for closing speed and TTC; warning at TTC < 3 s or < 5 m closing, centre band 30-70% of width. All thresholds provisional. | Decided |
| 16 | Shared detector order | Stage 1 trains pothole classes only (RDD2022 has no signs). Signs are added once a sign dataset is chosen, then the model is retrained on all classes. | Decided |
