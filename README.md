# Autonomous Road Anomaly Detection for Two-Wheelers

B.Tech project, Electronics & Communication Engineering, Delhi Technological University.
Team: Divyansh Kulshrestha, Harsh Vats, Daksh Kawatra. Supervisor: Prof. S. Indu.

A low-cost rider assistance system that runs on an Android phone mounted on a two-wheeler. It detects road potholes, recognises Indian road signs (including partly occluded ones), estimates the distance and speed of nearby vehicles, and detects crash-like events from inertial data, sending an emergency SMS with live location after a confirmation window.

All core inference runs on the device.

## Modules

| Module | Folder | Owner |
|---|---|---|
| 1. Pothole and crack detection | `detector/`, `app/.../pothole` | TBD |
| 2. Road sign detection (with occlusion) | `detector/`, `app/.../signs` | TBD |
| 3. Driving assistance (vehicle distance and speed) | `app/.../distance` | TBD |
| 4. Accident detection | `app/.../accident` | TBD |
| Shared pipeline and alerts | `app/.../pipeline`, `app/.../alerts` | TBD |

## Repository layout

- `detector/`: Python. Dataset configs, training, TFLite export.
- `app/`: Android app (Kotlin). One package per module, plus `pipeline/` and `alerts/`.
- `eval/`: Python. Evaluation scripts and result tables.
- `data/`: Not stored in git. See `data/README.md`.
- `docs/`: Decision log and notes.

## Setup

See `detector/README.md` for the Python environment and `app/README.md` for the Android build.

## Branching

- `main` is protected in spirit: merge only working code.
- One branch per module, e.g. `pothole/…`, `signs/…`, `distance/…`, `accident/…`, `pipeline/…`.
- Open a pull request into `main` and get one teammate to review.
