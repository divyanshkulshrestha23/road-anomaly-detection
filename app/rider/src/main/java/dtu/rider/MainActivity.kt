package dtu.rider

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Toast
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
    private var scheduler: FrameScheduler? = null
    private val potholeWarning = PotholeWarning()
    private lateinit var alerts: AlertManager
    private lateinit var sensors: SensorLogger
    private lateinit var crash: CrashDetector

    private var fpsEma = 0f
    private var lastFrameNs = 0L

    private val permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
        if (res[Manifest.permission.CAMERA] == true) startCamera()
        else Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        sensors.start()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraExecutor = Executors.newSingleThreadExecutor()
        alerts = AlertManager(binding.alert)
        sensors = SensorLogger(this)
        crash = CrashDetector(sensors, { state, msg ->
            runOnUiThread {
                Log.i(TAG, "Crash detector: $state $msg")
                when (state) {
                    CrashDetector.State.CONFIRMING, CrashDetector.State.ALERT ->
                        alerts.post(Alert(AlertLevel.CRASH, msg, "crash_$state"))
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
                Toast.makeText(this, "Saved to ${sensors.currentRideDir}", Toast.LENGTH_LONG).show()
            } else {
                val dir = sensors.startRecording()
                crash.recalibrate()
                binding.record.text = "Stop recording"
                Toast.makeText(this, "Recording to $dir", Toast.LENGTH_SHORT).show()
            }
        }

        distance = DistanceEstimator(this)
        tracker = VehicleTracker(distance)
        binding.snap.setOnClickListener { snapRequested = true }

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

        val needed = arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION)
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
            val preview = Preview.Builder().setResolutionSelector(selector).build()
                .also { it.setSurfaceProvider(binding.preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(cameraExecutor) { proxy ->
                val rotation = proxy.imageInfo.rotationDegrees
                val raw = proxy.toBitmap()
                proxy.close()
                val bmp = if (rotation == 0) raw else
                    Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, Matrix().apply { postRotate(rotation.toFloat()) }, true)
                onFrame(bmp)
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(bmp: Bitmap) {
        if (snapRequested) {
            snapRequested = false
            val f = sensors.saveSnapshot(bmp)
            runOnUiThread { Toast.makeText(this, "Saved ${f.name}", Toast.LENGTH_SHORT).show() }
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

        res.road?.let { dets ->
            lastRoad = dets
            if (potholeWarning.update(dets, w, h)) alerts.post(Alert(AlertLevel.POTHOLE, "POTHOLE AHEAD", "pothole"))
            sensors.logDetections(res.frameIndex, res.inferenceMs,
                dets.map { "${it.classId},${it.label},${"%.3f".format(it.score)},${it.box.left.toInt()},${it.box.top.toInt()},${it.box.right.toInt()},${it.box.bottom.toInt()}" },
                w, h)
        }
        res.vehicles?.let { vdets ->
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

        val speedKmh = sensors.lastSpeedMps?.let { "%.0f km/h".format(it * 3.6f) } ?: "no GPS"
        val gpu = if (detector?.usingGpu == true) "GPU" else "CPU"
        runOnUiThread {
            binding.stats.text = "FPS %.1f | road %d ms | veh %d ms | %s | %s %s%s".format(
                fpsEma, res.inferenceMs, res.vehicleMs, gpu, speedKmh,
                if (sensors.isRecording) "| REC" else "", if (distance.calibrated) "" else " | dist uncalibrated")
        }
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
