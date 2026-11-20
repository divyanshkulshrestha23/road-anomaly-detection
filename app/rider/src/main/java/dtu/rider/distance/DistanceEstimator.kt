package dtu.rider.distance

import android.content.Context
import android.graphics.RectF
import android.util.Log
import org.json.JSONObject
import kotlin.math.tan

/**
 * Monocular distance from a vehicle bounding box. Mirror of eval/distance_model.py: keep in sync.
 *
 *   width:  Z = W_real * f / w_px
 *   ground: Z = f * h_cam / (y_bottom - y_horizon)     (flat road)
 *   fused:  mean of both when they agree within 2x, otherwise the width estimate
 *
 * Calibration comes from assets/distance_calib.json (written by eval/distance_calibrate.py) and is
 * stored relative to the image's long side, so it holds in portrait and landscape:
 *   f_px = f_long * max(w, h);   horizon_y = h/2 + horizon_offset * max(w, h)
 * Without it, placeholder values are used and distances are only rough.
 */
class DistanceEstimator(context: Context) {
    var fLong = (0.5 / tan(Math.toRadians(66.0 / 2))).toFloat(); private set
    var horizonOffset = 0f; private set
    var cameraHeightM = 1.0f; private set
    var calibrated = false; private set
    private val widths = mutableMapOf("car" to 1.75f, "motorcycle" to 0.80f, "bicycle" to 0.60f, "bus" to 2.55f, "truck" to 2.50f)

    init {
        try {
            val j = JSONObject(context.assets.open("distance_calib.json").bufferedReader().readText())
            fLong = j.getDouble("f_long").toFloat()
            horizonOffset = j.getDouble("horizon_offset").toFloat()
            cameraHeightM = j.getDouble("camera_height_m").toFloat()
            j.optJSONObject("widths")?.let { w -> w.keys().forEach { k -> widths[k] = w.getDouble(k).toFloat() } }
            calibrated = j.optBoolean("calibrated", true)
        } catch (t: Throwable) {
            Log.w("DistanceEstimator", "No distance_calib.json; using placeholder calibration")
        }
    }

    fun isVehicle(label: String) = widths.containsKey(label)

    fun focalPx(imgW: Int, imgH: Int) = fLong * maxOf(imgW, imgH)

    private fun horizonPx(imgW: Int, imgH: Int) = imgH / 2f + horizonOffset * maxOf(imgW, imgH)

    fun widthDistance(label: String, box: RectF, imgW: Int, imgH: Int): Float? {
        val w = widths[label] ?: return null
        if (box.width() <= 1f) return null
        return w * focalPx(imgW, imgH) / box.width()
    }

    fun groundDistance(box: RectF, imgW: Int, imgH: Int): Float? {
        val dy = box.bottom - horizonPx(imgW, imgH)
        if (dy <= 2f) return null
        return focalPx(imgW, imgH) * cameraHeightM / dy
    }

    fun distance(label: String, box: RectF, imgW: Int, imgH: Int): Float? {
        val zw = widthDistance(label, box, imgW, imgH)
        val zg = groundDistance(box, imgW, imgH)
        if (zw == null) return zg
        if (zg == null || zg < 0.5f * zw || zg > 2f * zw) return zw
        return 0.5f * (zw + zg)
    }
}
