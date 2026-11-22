# First phone test (Harsh, 22 Nov 2026)

Phone: Samsung Galaxy S23 FE (SM-S711B), Android 16 (API 36). App in portrait, 480x640 analysis frames.
Inputs: two screen recordings (indoor + rides) and ride `ride_20261122_143816` (83 s, max 47 km/h).

## Measured
| Item | Value |
|---|---|
| Overall FPS (road model every frame, vehicle model every 2nd) | 10.9 (ride log); 10-13.7 on screen |
| Road model inference (GPU) | median 51 ms, 95th pct 62 ms |
| Vehicle model inference (GPU) | ~45-71 ms on frames where it runs |
| IMU rate delivered | **500 Hz** (50 Hz requested) |
| GPS | 1 Hz, no gaps, median accuracy 2 m (ride); speed stuck at 1-2 km/h in the second video session |
| Max acceleration on ride | 1.55 g (no impacts above the 2 g crash threshold) |
| Crash rule replay | no false alarm |

## Problems found and fixes (commit after this file)
1. Rider's own bike detected as "motorcycle ~1 m" (23% of vehicle detections) -> false VEHICLE AHEAD 1 m.
   Fix: drop vehicle boxes with bottom > 90% of frame height, width > 35%, centred.
2. Sign false positives on large close-up objects (keyboard: HOSPITAL; probably the round speedometer:
   persistent REGULATORY ~0 m). Real road signs in the ride were 2-8% of frame width, upper half.
   Fix: signs must be <= 25% of frame width and end above 85% of frame height; sign score >= 0.55.
3. Stale GPS kept signs on screen. Fix: fixes older than 5 s are ignored ("GPS stale" in status bar);
   panel shows "(passing)" instead of "~0 m".
4. IMU at 500 Hz broke the crash detector's timing assumptions. Fix: app keeps ~50 Hz (>= 18 ms apart);
   analysis replay does the same.
5. Vehicle tracks fragmented (123 of 205 tracks lived 1-2 updates) and closing speeds noisy (+-20 m/s).
   Fix: match by overlap or nearby centre of similar size; closing speed clamped to +-25 m/s;
   TTC and warnings only after 4 updates.
6. Testers expected a video from Record: Record saves sensor/detection data only. Toast now says so;
   use the phone's screen recorder for demo videos.

## Still below target
FPS 10.9 vs 15 target. Options: vehicle model every 3rd frame, smaller input (e.g. 480 px export,
no retraining), fewer per-frame allocations. Decide after re-measuring with the fixes.
