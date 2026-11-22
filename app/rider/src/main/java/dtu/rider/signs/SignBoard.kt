package dtu.rider.signs

import android.graphics.RectF
import android.location.Location
import dtu.rider.detect.Detection

/**
 * Keeps recently seen road signs on screen until the rider has passed them (decision #5).
 *
 * A sign is accepted after it is seen in [minHits] of the last [window] road frames.
 * Its distance ahead is estimated from the box width, assuming a sign face of [signWidthM].
 * It is removed once GPS shows the rider has travelled past that distance plus [passMarginM],
 * or after [maxShowMs] (also the fallback when there is no GPS fix).
 */
class SignBoard(
    private val signLabels: Set<String> = setOf("regulatory", "warning", "hospital", "school", "speed_limit"),
    private val minScore: Float = 0.55f,
    private val maxWidthFrac: Float = 0.25f,   // real signs seen from the bike are small (ride data: 2-8%)
    private val maxBottomFrac: Float = 0.85f,  // and never at the bottom of the frame (dashboard, speedometer)
    private val window: Int = 3,
    private val minHits: Int = 2,
    private val signWidthM: Float = 0.75f,
    private val passMarginM: Float = 10f,
    private val maxShowMs: Long = 30_000L,
    private val noGpsShowMs: Long = 12_000L,
) {
    data class ActiveSign(
        val label: String,
        val firstSeenMs: Long,
        val startLocation: Location?,
        var aheadM: Float,
        var lastSeenMs: Long,
    ) {
        /** Metres left to the sign, or null if unknown. */
        fun remaining(now: Location?): Float? {
            val s = startLocation ?: return null
            val n = now ?: return null
            return aheadM - s.distanceTo(n)
        }
    }

    private val history = ArrayDeque<Set<String>>()
    private val active = LinkedHashMap<String, ActiveSign>()

    fun isSign(d: Detection) = d.label in signLabels

    /** Rejects close-up look-alikes (keyboard keys, the bike's own round speedometer). */
    fun plausible(d: Detection, imgW: Int, imgH: Int) =
        d.box.width() <= maxWidthFrac * imgW && d.box.bottom <= maxBottomFrac * imgH

    /** Call on every road-model frame. Returns signs that became active on this frame. */
    fun update(dets: List<Detection>, imgW: Int, imgH: Int, focalPx: Float, location: Location?, nowMs: Long): List<ActiveSign> {
        val seen = dets.filter { isSign(it) && it.score >= minScore && plausible(it, imgW, imgH) }
        history.addLast(seen.map { it.label }.toSet())
        while (history.size > window) history.removeFirst()
        val fresh = mutableListOf<ActiveSign>()
        for (label in seen.map { it.label }.toSet()) {
            val hits = history.count { label in it }
            val box = seen.filter { it.label == label }.maxByOrNull { it.box.width() }!!.box
            val ahead = estimateAhead(box, focalPx)
            val cur = active[label]
            if (cur != null) {
                cur.lastSeenMs = nowMs
                // While still visible, refresh the distance estimate from the latest box.
                if (location != null && cur.startLocation != null) cur.aheadM = ahead + cur.startLocation.distanceTo(location)
                else cur.aheadM = ahead
            } else if (hits >= minHits) {
                val s = ActiveSign(label, nowMs, location?.let { Location(it) }, ahead, nowMs)
                active[label] = s
                fresh += s
            }
        }
        prune(location, nowMs)
        return fresh
    }

    fun prune(location: Location?, nowMs: Long) {
        val it = active.values.iterator()
        while (it.hasNext()) {
            val s = it.next()
            val rem = s.remaining(location)
            val age = nowMs - s.firstSeenMs
            val passed = rem != null && rem < -passMarginM
            val timeout = age > maxShowMs || (rem == null && nowMs - s.lastSeenMs > noGpsShowMs)
            if (passed || timeout) it.remove()
        }
    }

    fun current(): List<ActiveSign> = active.values.toList()

    private fun estimateAhead(box: RectF, focalPx: Float): Float =
        if (box.width() <= 1f) 50f else (signWidthM * focalPx / box.width()).coerceIn(2f, 150f)

    companion object {
        fun pretty(label: String) = when (label) {
            "speed_limit" -> "SPEED LIMIT"
            "school" -> "SCHOOL AHEAD"
            "hospital" -> "HOSPITAL AHEAD"
            "warning" -> "WARNING SIGN"
            "regulatory" -> "REGULATORY SIGN"
            else -> label.uppercase()
        }
    }
}
