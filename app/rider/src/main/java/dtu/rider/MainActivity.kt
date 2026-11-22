package dtu.rider

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.content.res.Configuration
import android.os.Build
import android.os.Bundle
import android.view.Surface
import android.util.Log
import android.text.InputType
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import dtu.rider.accident.CrashDetector
import dtu.rider.alerts.Alert
import dtu.rider.alerts.AlertLevel
import dtu.rider.alerts.AlertManager
import dtu.rider.databinding.ActivityMainBinding
import dtu.rider.detect.Detection
import dtu.rider.detect.YoloTfliteDetector
import dtu.rider.distance.CollisionWarning
import dtu.rider.distance.DistanceEstimator
import dtu.rider.distance.VehicleTracker
import dtu.rider.emergency.EmergencySms
import dtu.rider.signs.SignBoard
import dtu.rider.util.CrashReporter
import dtu.rider.util.PublicStorage
import dtu.rider.pipeline.FrameScheduler
import dtu.rider.pipeline.PotholeWarning
import dtu.rider.sensors.SensorLogger
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var detector: YoloTfliteDetector? = null
    private var vehicleDetector: YoloTfliteDetector? = null
    private lateinit var distance: DistanceEstimator
    private lateinit var tracker: VehicleTracker
    private val collision = CollisionWarning()
    @Volatile private var lastTracks: List<VehicleTracker.Track> = emptyList()
    @Volatile private var lastRoad: List<Detection> = emptyList()
    @Volatile private var snapRequested = false
    @Volatile private var frameErrorShown = false
    private val signBoard = SignBoard()
    private lateinit var sms: EmergencySms

    private val smsPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        Toast.makeText(this, if (ok) "SMS permission granted" else "SMS permission denied: crash SMS will not be sent", Toast.LENGTH_LONG).show()
    }
    private var scheduler: FrameScheduler? = null
    private val potholeWarning = PotholeWarning()
    private lateinit var alerts: AlertManager
    private lateinit var sensors: SensorLogger
    private lateinit var crash: CrashDetector

    private var preview: Preview? = null
    private var analysis: ImageAnalysis? = null
    private var fpsEma = 0f
    private var lastFrameNs = 0L

    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
        if (res[Manifest.permission.CAMERA] == true) startCamera()
        else Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        sensors.start()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        CrashReporter.install(this)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraExecutor = Executors.newSingleThreadExecutor()
        alerts = AlertManager(binding.alert)
        sensors = SensorLogger(this)
        sms = EmergencySms(this)
        crash = CrashDetector(sensors, { state, msg ->
            runOnUiThread {
                Log.i(TAG, "Crash detector: $state $msg")
                when (state) {
                    CrashDetector.State.CONFIRMING -> alerts.post(Alert(AlertLevel.CRASH, msg, "crash_$state"))
                    CrashDetector.State.ALERT -> {
                        val err = sms.send(sensors.lastLocation)
                        val text = if (err == null) "Emergency SMS sent to ${sms.contact}" else "Emergency SMS NOT sent: $err"
                        alerts.post(Alert(AlertLevel.CRASH, text, "crash_sms"))
                        Toast.makeText(this, text, Toast.LENGTH_LONG).show()
                    }
                    else -> Unit
                }
            }
        })
        sensors.addListener(crash)
        binding.alert.setOnClickListener { crash.cancel() }

        binding.record.setOnClickListener {
            if (sensors.isRecording) {
                sensors.stopRecording()
                binding.record.text = "Start recording"
                val dir = sensors.currentRideDir
                if (dir != null) Thread {
                    val msg = try { "Ride saved to ${PublicStorage.exportZip(this, dir)}" }
                              catch (t: Throwable) { Log.e(TAG, "Export failed", t); "Ride saved in app folder only (${t.message})" }
                    runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }
                }.start()
            } else {
                val dir = sensors.startRecording()
                crash.recalibrate()
                binding.record.text = "Stop recording"
                Toast.makeText(this, "Recording sensor + detection data (no video). Saved to Download/RiderAssist on Stop.", Toast.LENGTH_LONG).show()
            }
        }

        distance = DistanceEstimator(this)
        tracker = VehicleTracker(distance)
        binding.snap.setOnClickListener { snapRequested = true }
        binding.sos.setOnClickListener { showSosDialog() }

        val problems = mutableListOf<String>()
        try {
            detector = YoloTfliteDetector(this)
        } catch (t: Throwable) {
            Log.e(TAG, "Road detector failed to load", t)
            problems += "No road model (assets/detector.tflite): ${t.message}"
        }
        try {
            vehicleDetector = YoloTfliteDetector(this, modelAsset = "vehicle.tflite", labelsAsset = "vehicle_labels.txt", scoreThreshold = 0.4f)
        } catch (t: Throwable) {
            Log.e(TAG, "Vehicle detector failed to load", t)
            problems += "No vehicle model (assets/vehicle.tflite): ${t.message}"
        }
        if (detector != null || vehicleDetector != null) scheduler = FrameScheduler(detector, vehicleDetector)
        if (!distance.calibrated) problems += "Distance not calibrated (assets/distance_calib.json)"
        if (problems.isNotEmpty()) binding.stats.text = problems.joinToString("\n")

        val needed = mutableListOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION).apply {
            if (Build.VERSION.SDK_INT < 29) add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }.toTypedArray()
        if (needed.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            startCamera(); sensors.start()
        } else permissionLauncher.launch(needed)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
            val rotation = binding.root.display?.rotation ?: Surface.ROTATION_0
            val preview = Preview.Builder().setResolutionSelector(selector).setTargetRotation(rotation).build()
                .also { it.setSurfaceProvider(binding.preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setTargetRotation(rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            this.preview = preview
            this.analysis = analysis
            analysis.setAnalyzer(cameraExecutor) { proxy ->
                val rotation = proxy.imageInfo.rotationDegrees
                val raw = proxy.toBitmap()
                proxy.close()
                val bmp = if (rotation == 0) raw else
                    Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
                try {
                    onFrame(bmp)
                } catch (t: Throwable) {
                    // Keep running: show the error once instead of crashing on every frame.
                    Log.e(TAG, "Frame processing failed", t)
                    if (!frameErrorShown) {
                        frameErrorShown = true
                        runOnUiThread { binding.stats.text = "Frame error: ${t.javaClass.simpleName}: ${t.message}" }
                    }
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(bmp: Bitmap) {
        if (snapRequested) {
            snapRequested = false
            val f = sensors.saveSnapshot(bmp)
            val msg = try { "Saved to ${PublicStorage.saveImage(this, bmp, f.nameWithoutExtension)}" }
                      catch (t: Throwable) { Log.e(TAG, "Public save failed", t); "Saved in app folder only: ${f.name}" }
            runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
        }
        val sch = scheduler ?: return
        val res = sch.process(bmp)
        val now = System.nanoTime()
        if (lastFrameNs != 0L) {
            val fps = 1e9f / (now - lastFrameNs)
            fpsEma = if (fpsEma == 0f) fps else 0.9f * fpsEma + 0.1f * fps
        }
        lastFrameNs = now
        val w = bmp.width; val h = bmp.height

        val gps = sensors.freshLocation()
        res.road?.let { raw ->
            // Hide implausible sign boxes (speedometer, close-ups) from the overlay as well.
            val dets = raw.filter { !signBoard.isSign(it) || signBoard.plausible(it, w, h) }
            lastRoad = dets
            val fresh = signBoard.update(dets, w, h, distance.focalPx(w, h), gps, System.currentTimeMillis())
            fresh.forEach { alerts.post(Alert(AlertLevel.SIGN, SignBoard.pretty(it.label), "sign_${it.label}")) }
            val signText = signBoard.current().joinToString("\n") { sgn ->
                val rem = sgn.remaining(gps)
                SignBoard.pretty(sgn.label) + when {
                    rem == null -> ""
                    rem <= 2f -> "  (passing)"
                    else -> "  ~%.0f m".format(rem)
                }
            }
            runOnUiThread {
                binding.signs.text = signText
                binding.signs.visibility = if (signText.isEmpty()) android.view.View.GONE else android.view.View.VISIBLE
            }
            if (potholeWarning.update(dets, w, h)) alerts.post(Alert(AlertLevel.POTHOLE, "POTHOLE AHEAD", "pothole"))
            sensors.logDetections(res.frameIndex, res.inferenceMs,
                dets.map { "${it.classId},${it.label},${"%.3f".format(it.score)},${it.box.left.toInt()},${it.box.top.toInt()},${it.box.right.toInt()},${it.box.bottom.toInt()}" },
                w, h)
        }
        res.vehicles?.let { all ->
            // Drop the rider's own bike: wide box at the bottom-centre of the frame (23% of detections in the first ride).
            val vdets = all.filter { d ->
                val ego = d.box.bottom > 0.9f * h && d.box.width() > 0.35f * w && d.box.centerX() in 0.2f * w..0.8f * w
                !ego
            }
            val tracks = tracker.update(vdets, w, h, System.currentTimeMillis())
            lastTracks = tracks
            collision.check(tracks, w)?.let { t ->
                alerts.post(Alert(AlertLevel.COLLISION, "VEHICLE AHEAD %.0f m".format(t.z), "collision"))
            }
            sensors.logVehicles(res.frameIndex, tracks.map { t ->
                "${t.id},${t.label},${"%.2f".format(t.z)},${"%.2f".format(t.closingMps)},${t.ttcS?.let { "%.2f".format(it) } ?: ""}," +
                    "${t.box.left.toInt()},${t.box.top.toInt()},${t.box.right.toInt()},${t.box.bottom.toInt()}"
            }, w, h)
        }

        // Overlay: road detections plus tracked vehicles labelled with distance and TTC.
        val vehicleBoxes = lastTracks.map { t ->
            val ttc = t.ttcS?.let { " TTC %.1fs".format(it) } ?: ""
            Detection(100, "${t.label} %.0fm$ttc".format(t.z), 1f, t.box)
        }
        binding.overlay.update(lastRoad + vehicleBoxes, w, h, potholeWarning.pathZone(w, h))

        val speedKmh = when {
            gps == null && sensors.lastLocation != null -> "GPS stale"
            gps == null -> "no GPS"
            else -> sensors.lastSpeedMps?.let { "%.0f km/h".format(it * 3.6f) } ?: "no speed"
        }
        val gpu = if (detector?.usingGpu == true) "GPU" else "CPU"
        runOnUiThread {
            binding.stats.text = "FPS %.1f | road %d ms | veh %d ms | %s | %s %s%s".format(
                fpsEma, res.inferenceMs, res.vehicleMs, gpu, speedKmh,
                if (sensors.isRecording) "| REC" else "", if (distance.calibrated) "" else " | dist uncalibrated")
        }
    }

    /** Portrait and landscape both work: the activity is not recreated, only the camera rotation updates. */
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        val rotation = binding.root.display?.rotation ?: Surface.ROTATION_0
        preview?.targetRotation = rotation
        analysis?.targetRotation = rotation
        // The phone's "upright" changed, so the crash detector's tilt reference must be re-learned (~3 s).
        crash.recalibrate()
    }

    private fun showSosDialog() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_PHONE
            hint = "+91XXXXXXXXXX"
            setText(sms.contact ?: "")
        }
        AlertDialog.Builder(this)
            .setTitle("Emergency contact")
            .setMessage("Gets an SMS with your location if a crash is detected and you don't cancel.")
            .setView(input)
            .setPositiveButton("Save") { _, _ ->
                sms.contact = input.text.toString()
                if (!sms.hasPermission()) smsPermission.launch(Manifest.permission.SEND_SMS)
                Toast.makeText(this, "Saved ${sms.contact ?: "(none)"}", Toast.LENGTH_SHORT).show()
            }
            .setNeutralButton("Send test SMS") { _, _ ->
                sms.contact = input.text.toString()
                if (!sms.hasPermission()) { smsPermission.launch(Manifest.permission.SEND_SMS); return@setNeutralButton }
                val err = sms.send(sensors.lastLocation, test = true)
                Toast.makeText(this, err ?: "Test SMS sent to ${sms.contact}", Toast.LENGTH_LONG).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        sensors.stop()
        cameraExecutor.shutdown()
        detector?.close()
        alerts.release()
    }

    companion object { private const val TAG = "RiderAssist" }
}
