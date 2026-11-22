package dtu.rider.distance

/**
 * Raises a forward-collision warning for a tracked vehicle in the rider's lane:
 * box centre in the middle band of the frame, seen for a few frames, and either
 * time-to-collision below [ttcWarnS] or closer than [minGapM] while closing.
 * Thresholds are provisional; tune on recorded rides.
 */
class CollisionWarning(
    private val laneLeft: Float = 0.30f,
    private val laneRight: Float = 0.70f,
    private val ttcWarnS: Float = 3.0f,
    private val minGapM: Float = 5.0f,
    private val minHits: Int = 4,
) {
    fun check(tracks: List<VehicleTracker.Track>, imgW: Int): VehicleTracker.Track? =
        tracks.filter { t ->
            val cx = t.box.centerX() / imgW
            cx in laneLeft..laneRight && t.hits >= minHits &&
                ((t.ttcS ?: Float.MAX_VALUE) < ttcWarnS || (t.z < minGapM && t.closingMps > 0.3f))
        }.minByOrNull { it.ttcS ?: it.z }
}
