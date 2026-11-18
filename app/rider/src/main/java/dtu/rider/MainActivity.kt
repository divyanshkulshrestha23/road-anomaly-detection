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
import dtu.rider.detect.YoloTfliteDetector
import dtu.rider.pipeline.FrameScheduler
import dtu.rider.pipeline.PotholeWarning
import dtu.rider.sensors.SensorLogger
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var detector: YoloTfliteDetector? = null
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

        try {
            detector = YoloTfliteDetector(this).also { scheduler = FrameScheduler(it) }
        } catch (t: Throwable) {
            Log.e(TAG, "Detector failed to load", t)
            binding.stats.text = "No model: copy detector.tflite into rider/src/main/assets/\n${t.message}"
        }

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
        val sch = scheduler ?: return
        val res = sch.process(bmp)
        val now = System.nanoTime()
        if (lastFrameNs != 0L) {
            val fps = 1e9f / (now - lastFrameNs)
            fpsEma = if (fpsEma == 0f) fps else 0.9f * fpsEma + 0.1f * fps
        }
        lastFrameNs = now

        val dets = res.road ?: return
        val zone = potholeWarning.pathZone(bmp.width, bmp.height)
        if (potholeWarning.update(dets, bmp.width, bmp.height)) {
            alerts.post(Alert(AlertLevel.POTHOLE, "POTHOLE AHEAD", "pothole"))
        }
        sensors.logDetections(res.frameIndex, res.inferenceMs,
            dets.map { "${it.classId},${it.label},${"%.3f".format(it.score)},${it.box.left.toInt()},${it.box.top.toInt()},${it.box.right.toInt()},${it.box.bottom.toInt()}" },
            bmp.width, bmp.height)
        binding.overlay.update(dets, bmp.width, bmp.height, zone)
        val speedKmh = sensors.lastSpeedMps?.let { "%.0f km/h".format(it * 3.6f) } ?: "no GPS"
        runOnUiThread {
            binding.stats.text = "FPS %.1f | infer %d ms | %s | %s | %s".format(
                fpsEma, res.inferenceMs, if (detector?.usingGpu == true) "GPU" else "CPU", speedKmh,
                if (sensors.isRecording) "REC" else "")
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
