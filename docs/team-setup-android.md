# Rider Assist: status, setup and testing guide

For teammates setting up the Android app for the first time. Last updated 20 Nov 2026.

---

## 1. What the app does right now

Rider Assist runs on an Android phone mounted on a two-wheeler. Everything runs on the phone; no internet is needed.

| Part | Status |
|---|---|
| **Module 1: Potholes and cracks** | Model trained (RDD2022, 138 epochs). Test mAP@0.5 = 0.555 overall, 0.364 on Indian roads. Draws boxes on the camera view; beeps "POTHOLE AHEAD" when a pothole is in the rider's path in 2 of the last 3 frames. |
| **Module 2: Road signs** | Model training on Kaggle now (potholes + signs together, with partly hidden signs). Display code is in the app (green panel, removed once you pass the sign); it activates when the new model is added. |
| **Module 3: Vehicle distance** | Done in code. Detects cars, motorcycles, buses, trucks and bicycles, estimates distance and time-to-collision, warns "VEHICLE AHEAD". **Not calibrated yet**, so distances are rough. |
| **Module 4: Accident detection** | Crash rule in code (tilt + speed drop + stillness, then a 20 s cancel countdown), then an automatic SMS with a map link. Set the number with the **SOS contact** button. |
| **Recording** | Record button saves IMU, GPS, detections and vehicle tracks to CSV, for tuning and the report. |
| **Snap** | Saves the exact frame the app sees (used for distance calibration). |

The app has **not been tested on a phone yet**. The first goal is to get it built and running.

## 2. Work still left (review on 29-30 Nov)

1. Build the app and test it on a phone (this guide).
2. Distance calibration photos with the Snap button (Module 3).
3. Test rides with recording on (Module 4 tuning, sensor rate).
4. Put the signs model into the app when Kaggle training finishes; show sign labels until the rider passes them.
5. Test the SOS SMS (SOS contact → Send test SMS) and tune the crash rule on recordings.
6. Measure FPS on the phone with all models running.
7. Update slides with results; check references and accident statistics.
8. 27-28 Nov: freeze features, rehearse the demo.

---

## 3. Install Android Studio

**The app is Android only. It will not install on an iPhone.** Use an Android phone, and ideally the same phone the team will use for the demo.

1. Download Android Studio (current stable) from the official Android developer site: https://developer.android.com/studio
2. Install with the default options.
3. On first launch, the setup wizard downloads the Android SDK. Choose **Standard** and accept the licences. This can take 10-20 minutes.

## 4. Get the code and open it

1. In Android Studio's welcome screen: **Clone Repository** (or **Get from VCS**), URL:
   `https://github.com/divyanshkulshrestha23/road-anomaly-detection`
2. If it opens the whole repo, close it: **File → Close Project**.
3. **File → Open** and select the **`app`** folder inside `road-anomaly-detection`. (The Android project is `app/`, not the repo root.) Click **Trust Project** if asked.
4. Wait for **Gradle sync** to finish (green tick in the **Build** tab at the bottom). The first time takes about 15 minutes. Keep the laptop awake and plugged in.
   - If it offers to **upgrade the Android Gradle Plugin**, say **no**.
   - If you see a **JDK / Java version** error: **File → Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK**, pick the bundled one (e.g. "jbr-17" or "jbr-21"), then click the **Sync** (elephant) icon.

Before working each day: **Git → Pull**, so everyone runs the same code.

## 5. Add the model files

The model files are **not in the repo**. Get these three files from Divyansh (Drive/WhatsApp):

- `detector.tflite` (pothole/crack model, 10.7 MB)
- `vehicle.tflite` (vehicle model, 10.8 MB)
- `labels.txt`

Put them in `app/rider/src/main/assets/`:
1. In the left panel, switch the dropdown at the top from **Android** to **Project**.
2. Open `app → rider → src → main → assets`.
3. Right-click **assets → Open In → Explorer** (Finder on Mac) and paste the files. Replace `labels.txt` if asked.

## 6. Build

**Build → Make Project** (Ctrl+F9). Wait for **BUILD SUCCESSFUL** in the Build tab.

If you get red errors, copy **all** the red lines and send them to Divyansh. Don't try to fix them yourself; fixes go into the repo and you just **Git → Pull** and build again.

## 7. Install on the phone

**Option A: USB cable (preferred for testing, gives logs)**

1. On the phone: **Settings → About phone →** tap **Build number** 7 times (on Xiaomi/Realme/Oppo it may be "MIUI version" or "OS version").
2. **Developer options** (often under Settings → System or Additional settings) → turn on **USB debugging**.
3. Connect a USB cable that carries data. Tap **Allow** on the "Allow USB debugging?" prompt.
4. In Android Studio, pick the phone in the device dropdown at the top, then click the green **▶ Run**.
5. Allow **camera** and **location** on the phone.

**Option B: APK file (no cable, but no logs if it crashes)**

1. **Build → Build App Bundle(s) / APK(s) → Build APK(s)** (newer versions: **Build → Generate App Bundles or APKs → Generate APKs**).
2. The file is at `app\rider\build\outputs\apk\debug\rider-debug.apk`. Share it via Google Drive.
3. On the phone, open it and allow "Install unknown apps" for the app you opened it from. If Play Protect warns, choose **More details → Install anyway**.

## 8. Testing

Go in this order. Note what you see at each step.

**Step 1: First launch (desk, 5 min)**
- The status bar at the top should show **FPS**, **road ... ms**, **veh ... ms**, **GPU or CPU**, and GPS speed. Write these numbers down.
- "dist uncalibrated" is expected for now.
- "No road model" / "No vehicle model" means the files in step 5 are missing.
- Tap **Snap**: should show "Saved snap_...jpg".
- Tap **Start recording**, wait 30 s, **Stop**: should show where it saved.
- Tap **SOS contact**, enter your own number, tap **Send test SMS**, and allow SMS permission. You should receive a "TEST from Rider Assist" SMS with a map link.

**Step 2: Indoor check (5 min)**
- Point the phone at a **pothole photo** on a laptop screen: expect a red "pothole" box.
- Point it at a **traffic photo**: cars and bikes get blue boxes with a distance (e.g. "car 12m").
- This only checks that the models load and draw; it doesn't need to be accurate.

**Step 3: Outdoors, standing still (10 min)**
- Hold the phone in landscape at about handlebar height by a road.
- Do potholes get boxes? Do passing vehicles get boxes and distances? Does GPS speed appear?
- Leave it running 5 minutes in daylight and note whether FPS drops (the phone heating up).

**Step 4: Short ride (only after steps 1-3 work)**
- Mount the phone **securely** in landscape. Press **Start recording**, ride 5-10 min on a familiar road with some potholes and one hard brake, then **Stop**.
- **Do not look at the screen while riding.** Check afterwards, or have a passenger watch.
- **No staged falls.** A safe crash-rule test is laying the parked bike down slowly with the app running.

**Send Divyansh:**
- FPS, road ms, veh ms, GPU/CPU, and the phone model.
- Screenshots of anything wrong (wrong boxes, boxes in the wrong place, odd distances).
- The recordings folder, zipped: `Android/data/dtu.rider/files/recordings/` (copy with a cable, or Android Studio **View → Tool Windows → Device Explorer**).
- If the app **crashes**: connect by USB, open **Logcat** at the bottom of Android Studio, type `RiderAssist` in the filter, and send the red lines.
