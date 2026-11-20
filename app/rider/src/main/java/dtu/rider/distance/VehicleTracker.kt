package dtu.rider.distance

import android.graphics.RectF
import dtu.rider.detect.Detection
import kotlin.math.max
import kotlin.math.min

/**
 * Tracks vehicles across frames (greedy IoU matching) and smooths each one's distance with an
 * alpha-beta filter, giving distance Z, closing speed and time-to-collision (TTC).
 * Closing speed > 0 means the gap is shrinking.
 */
class VehicleTracker(
    private val estimator: DistanceEstimator,
    private val iouMatch: Float = 0.3f,
    private val maxMissMs: Long = 1000L,
    private val alpha: Float = 0.4f,
    private val beta: Float = 0.1f,
) {
    data class Track(
        val id: Int,
        var label: String,
        var box: RectF,
        var z: Float,
        var zRate: Float = 0f,        // m/s, negative when approaching
        var lastMs: Long,
        var hits: Int = 1,
    ) {
        val closingMps get() = -zRate
        val ttcS: Float? get() = if (closingMps > 0.5f) z / closingMps else null
    }

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    fun update(dets: List<Detection>, imgW: Int, imgH: Int, nowMs: Long): List<Track> {
        val vehicles = dets.filter { estimator.isVehicle(it.label) }
        val unmatched = vehicles.toMutableList()
        // Greedy matching, best IoU first.
        val pairs = mutableListOf<Triple<Track, Detection, Float>>()
        for (t in tracks) for (d in vehicles) {
            val i = iou(t.box, d.box); if (i >= iouMatch) pairs += Triple(t, d, i)
        }
        val usedT = mutableSetOf<Int>()
        for ((t, d, _) in pairs.sortedByDescending { it.third }) {
            if (t.id in usedT || d !in unmatched) continue
            val zMeas = estimator.distance(d.label, d.box, imgW, imgH) ?: continue
            val dt = ((nowMs - t.lastMs) / 1000f).coerceAtLeast(1e-3f)
            val zPred = t.z + t.zRate * dt
            val r = zMeas - zPred
            t.z = zPred + alpha * r
            t.zRate += beta * r / dt
            t.box = d.box; t.label = d.label; t.lastMs = nowMs; t.hits++
            usedT += t.id; unmatched -= d
        }
        for (d in unmatched) {
            val z = estimator.distance(d.label, d.box, imgW, imgH) ?: continue
            tracks += Track(nextId++, d.label, d.box, z, 0f, nowMs)
        }
        tracks.removeAll { nowMs - it.lastMs > maxMissMs }
        return tracks.filter { it.lastMs == nowMs }
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val iy = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val inter = ix * iy
        val u = a.width() * a.height() + b.width() * b.height() - inter
        return if (u <= 0f) 0f else inter / u
    }
}
