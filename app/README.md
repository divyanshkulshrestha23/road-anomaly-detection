# Android app (Kotlin): Rider Assist

Gradle project root is this folder (`app/`); the app module is `rider/`.

## Build and run

1. Install Android Studio (recent stable). Open the `app/` folder (not the repo root).
2. If Studio asks to create or update the Gradle wrapper, accept. The plugin versions in
   `build.gradle.kts` are a starting point; accept Studio's suggested upgrades if the sync fails.
   JDK 17 is required.
3. Copy the trained FP16 model to `rider/src/main/assets/detector.tflite`
   (from the Kaggle notebook's Output tab: `exports/*float16*.tflite`).
4. Enable USB debugging on the test phone, connect it, and press Run.

Without a model the app still starts and shows a message; sensors and recording still work.

## What it does today

- Rear camera preview with detection boxes (CameraX + TFLite, GPU delegate if available)
- Pothole warning when a pothole sits in the dashed path zone in 2 of the last 3 frames
- On-screen FPS, inference time, CPU/GPU, GPS speed
- **Record button**: logs IMU (~50 Hz requested), GPS (1 Hz) and every frame's detections to
  `Android/data/dtu.rider/files/recordings/ride_<time>/` (imu.csv, gps.csv, detections.csv, device.txt).
  Copy these off the phone for analysis in `eval/`.
- Crash detector state machine (multi-signal rule, see `docs/decisions.md` #3) with a cancel
  countdown; tap the red banner to cancel. SMS sending is not wired yet.

## Packages (`rider/src/main/java/dtu/rider/`)

- `pipeline/`: frame scheduler, pothole warning logic
- `detect/`: YOLO TFLite wrapper (letterbox, decode, NMS)
- `ui/`: overlay drawing
- `alerts/`: priority alert queue with cooldown and beep
- `sensors/`: IMU + GPS streaming and CSV recording
- `accident/`: crash rule
- Signs and vehicle distance plug into `FrameScheduler` once their models exist.

## Not compiled yet

This code was written without an Android SDK to hand. Expect a few fixes on the first build;
paste any build errors back so they can be fixed in the repo.
