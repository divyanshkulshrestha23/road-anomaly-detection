# Android app (Kotlin)

Android app that runs the camera pipeline, on-device models, sensors, and alerts.

## Packages

- `pipeline/`: camera frames in, frame scheduler out (shared by all vision modules)
- `pothole/`: pothole detector wrapper and overlay
- `signs/`: sign detector wrapper, occlusion handling, GPS-based label removal
- `distance/`: vehicle detector, monocular distance and speed estimate, time-to-collision
- `accident/`: inertial features, crash rule, confirmation window
- `alerts/`: priority alert queue, audio warnings, emergency SMS

## Status

Skeleton only. Build with Android Studio. Gradle project to be created on the first module branch.
