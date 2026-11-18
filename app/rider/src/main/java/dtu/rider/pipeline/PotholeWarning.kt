package dtu.rider.pipeline

import android.graphics.RectF
import dtu.rider.detect.Detection

/**
 * Decides when a pothole detection is worth warning about: it must sit inside the rider's path
 * zone (lower-centre of the frame) and be seen in at least [minHits] of the last [window] frames.
 */
class PotholeWarning(private val window: Int = 3, private val minHits: Int = 2, private val minScore: Float = 0.45f) {
    private val history = ArrayDeque<Boolean>()

    fun pathZone(w: Int, h: Int) = RectF(w * 0.25f, h * 0.55f, w * 0.75f, h.toFloat())

    /** Returns true when a warning should be raised for this frame. */
    fun update(dets: List<Detection>, w: Int, h: Int, potholeClassId: Int = 0): Boolean {
        val zone = pathZone(w, h)
        val hit = dets.any { it.classId == potholeClassId && it.score >= minScore && RectF.intersects(zone, it.box) }
        history.addLast(hit)
        while (history.size > window) history.removeFirst()
        return hit && history.count { it } >= minHits
    }
}
