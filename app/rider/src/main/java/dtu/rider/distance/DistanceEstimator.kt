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
 * Calibration comes from assets/distance_calib.json (written by eval/distance_calibrate.py).
 * Without it, placeholder values are used and distances are only rough.
 */
class DistanceEstimator(context: Context) {
    var fNorm = (0.5 / tan(Math.toRadians(66.0 / 2))).toFloat(); private set
    var horizonNorm = 0.5f; private set
    var cameraHeightM = 1.0f; private set
    var calibrated = false; private set
    private val widths = mutableMapOf("car" to 1.75f, "motorcycle" to 0.80f, "bicycle" to 0.60f, "bus" to 2.55f, "truck" to 2.50f)

    init {
        try {
            val j = JSONObject(context.assets.open("distance_calib.json").bufferedReader().readText())
            fNorm = j.getDouble("f_norm").toFloat()
            horizonNorm = j.getDouble("horizon_norm").toFloat()
            cameraHeightM = j.getDouble("camera_height_m").toFloat()
            j.optJSONObject("widths")?.let { w -> w.keys().forEach { k -> widths[k] = w.getDouble(k).toFloat() } }
            calibrated = j.optBoolean("calibrated", true)
        } catch (t: Throwable) {
            Log.w("DistanceEstimator", "No distance_calib.json; using placeholder calibration")
        }
    }

    fun isVehicle(label: String) = widths.containsKey(label)

    fun widthDistance(label: String, box: RectF, imgW: Int): Float? {
        val w = widths[label] ?: return null
        if (box.width() <= 1f) return null
        return w * fNorm * imgW / box.width()
    }

    fun groundDistance(box: RectF, imgW: Int, imgH: Int): Float? {
        val dy = box.bottom - horizonNorm * imgH
        if (dy <= 2f) return null
        return fNorm * imgW * cameraHeightM / dy
    }

    fun distance(label: String, box: RectF, imgW: Int, imgH: Int): Float? {
        val zw = widthDistance(label, box, imgW)
        val zg = groundDistance(box, imgW, imgH)
        if (zw == null) return zg
        if (zg == null || zg < 0.5f * zw || zg > 2f * zw) return zw
        return 0.5f * (zw + zg)
    }
}
