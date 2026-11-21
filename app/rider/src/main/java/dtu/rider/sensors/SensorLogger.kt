package dtu.rider.sensors

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.os.Looper
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Streams accelerometer + gyroscope (~50 Hz requested) and GPS (~1 Hz) to listeners, and
 * optionally to CSV files for offline analysis in Python (eval/).
 *
 * Files go to: Android/data/dtu.rider/files/recordings/ride_<timestamp>_{imu,gps,detections}.csv
 * Note: Android treats the sampling period as a hint; check the real rate from the timestamps.
 */
class SensorLogger(private val context: Context) : SensorEventListener {

    interface Listener {
        fun onImu(tNs: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float) {}
        fun onGps(tMs: Long, lat: Double, lon: Double, speedMps: Float?, accuracyM: Float?) {}
    }

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private val listeners = mutableListOf<Listener>()

    private val gyro = FloatArray(3)
    @Volatile var lastSpeedMps: Float? = null; private set
    @Volatile var lastLocation: Location? = null; private set

    private var imuOut: BufferedWriter? = null
    private var gpsOut: BufferedWriter? = null
    private var detOut: BufferedWriter? = null
    private var vehOut: BufferedWriter? = null
    val isRecording get() = imuOut != null
    var currentRideDir: File? = null; private set

    fun addListener(l: Listener) { listeners += l }

    // One lock for all CSV writers. An explicit lock (not `synchronized` inside an inline function)
    // because that combination produced bytecode the Android verifier rejected (VerifyError at startup).
    private val ioLock = java.util.concurrent.locks.ReentrantLock()

    /** Writes under the lock; ignores writes that race with Stop recording closing the file. */
    private fun safeWrite(w: BufferedWriter, block: (BufferedWriter) -> Unit) {
        ioLock.lock()
        try {
            block(w)
        } catch (_: java.io.IOException) {
        } finally {
            ioLock.unlock()
        }
    }

    private val locationListener = object : LocationListener {
        override fun onLocationChanged(loc: Location) {
            lastLocation = loc
            val speed = if (loc.hasSpeed()) loc.speed else null
            lastSpeedMps = speed
            listeners.forEach { it.onGps(loc.time, loc.latitude, loc.longitude, speed, if (loc.hasAccuracy()) loc.accuracy else null) }
            gpsOut?.let { w ->
                safeWrite(w) { it.write("${loc.time},${System.currentTimeMillis()},${loc.latitude},${loc.longitude},${speed ?: ""},${if (loc.hasAccuracy()) loc.accuracy else ""}\n") }
            }
        }
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
        @Deprecated("Required on API < 29")
        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    }

    @SuppressLint("MissingPermission")
    fun start() {
        val periodUs = 20_000 // 50 Hz requested
        sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let { sm.registerListener(this, it, periodUs) }
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(this, it, periodUs) }
        try {
            lm.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, locationListener, Looper.getMainLooper())
        } catch (e: SecurityException) {
            Log.w(TAG, "No location permission", e)
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "No GPS provider", e)
        }
    }

    fun stop() {
        sm.unregisterListener(this)
        lm.removeUpdates(locationListener)
        stopRecording()
    }

    fun startRecording(): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(context.getExternalFilesDir(null), "recordings/ride_$stamp").apply { mkdirs() }
        imuOut = BufferedWriter(FileWriter(File(dir, "imu.csv"))).apply { write("t_ns,t_wall_ms,ax,ay,az,gx,gy,gz\n") }
        gpsOut = BufferedWriter(FileWriter(File(dir, "gps.csv"))).apply { write("t_ms,t_wall_ms,lat,lon,speed_mps,accuracy_m\n") }
        detOut = BufferedWriter(FileWriter(File(dir, "detections.csv"))).apply {
            write("t_ms,frame,inference_ms,class_id,label,score,left,top,right,bottom,img_w,img_h\n")
        }
        vehOut = BufferedWriter(FileWriter(File(dir, "vehicles.csv"))).apply {
            write("t_ms,frame,track_id,label,distance_m,closing_mps,ttc_s,left,top,right,bottom,img_w,img_h,gps_speed_mps\n")
        }
        File(dir, "device.txt").writeText(
            "manufacturer=${android.os.Build.MANUFACTURER}\nmodel=${android.os.Build.MODEL}\n" +
                "soc=${if (android.os.Build.VERSION.SDK_INT >= 31) android.os.Build.SOC_MODEL else "unknown"}\n" +
                "android_sdk=${android.os.Build.VERSION.SDK_INT}\n"
        )
        currentRideDir = dir
        return dir
    }

    fun stopRecording() {
        listOfNotNull(imuOut, gpsOut, detOut, vehOut).forEach { w -> safeWrite(w) { it.flush(); it.close() } }
        imuOut = null; gpsOut = null; detOut = null; vehOut = null
    }

    /** Called from the camera thread with each frame's detections (or an empty row for timing). */
    fun logDetections(frame: Long, inferenceMs: Long, rows: List<String>, imgW: Int, imgH: Int) {
        val w = detOut ?: return
        val t = System.currentTimeMillis()
        safeWrite(w) { out ->
            if (rows.isEmpty()) out.write("$t,$frame,$inferenceMs,,,,,,,,$imgW,$imgH\n")
            else rows.forEach { out.write("$t,$frame,$inferenceMs,$it,$imgW,$imgH\n") }
        }
    }

    /** One row per tracked vehicle; [rows] are pre-formatted "id,label,z,closing,ttc,l,t,r,b". */
    fun logVehicles(frame: Long, rows: List<String>, imgW: Int, imgH: Int) {
        val w = vehOut ?: return
        val t = System.currentTimeMillis()
        val v = lastSpeedMps?.toString() ?: ""
        safeWrite(w) { out -> rows.forEach { out.write("$t,$frame,$it,$imgW,$imgH,$v\n") } }
    }

    /** Saves the exact frame the app analyses (same crop as detection) in the app folder; see PublicStorage for the visible copy. */
    fun saveSnapshot(bitmap: android.graphics.Bitmap, tag: String = "snap"): File {
        val dir = File(context.getExternalFilesDir(null), "snapshots").apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        val f = File(dir, "${tag}_$stamp.jpg")
        f.outputStream().use { bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, it) }
        return f
    }

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> { gyro[0] = e.values[0]; gyro[1] = e.values[1]; gyro[2] = e.values[2] }
            Sensor.TYPE_ACCELEROMETER -> {
                val (ax, ay, az) = Triple(e.values[0], e.values[1], e.values[2])
                listeners.forEach { it.onImu(e.timestamp, ax, ay, az, gyro[0], gyro[1], gyro[2]) }
                imuOut?.let { w -> safeWrite(w) { it.write("${e.timestamp},${System.currentTimeMillis()},$ax,$ay,$az,${gyro[0]},${gyro[1]},${gyro[2]}\n") } }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    companion object { private const val TAG = "SensorLogger" }
}
