package dtu.rider.pipeline

import android.graphics.Bitmap
import android.os.SystemClock
import dtu.rider.detect.Detection
import dtu.rider.detect.YoloTfliteDetector

/**
 * Decides which model runs on which frame. Today there is one shared road detector
 * (potholes + cracks, later signs). The vehicle detector for Module 3 plugs in at
 * [vehicleEvery] once it exists.
 */
class FrameScheduler(
    private val roadDetector: YoloTfliteDetector,
    private val roadEvery: Int = 1,
    private val vehicleEvery: Int = 3,
) {
    data class Result(
        val frameIndex: Long,
        val road: List<Detection>?,   // null when the road model was skipped this frame
        val inferenceMs: Long,
    )

    private var frame = 0L

    fun process(bitmap: Bitmap): Result {
        val idx = frame++
        var ms = 0L
        var road: List<Detection>? = null
        if (idx % roadEvery == 0L) {
            val t0 = SystemClock.elapsedRealtime()
            road = roadDetector.detect(bitmap)
            ms = SystemClock.elapsedRealtime() - t0
        }
        // TODO(Module 3): if (idx % vehicleEvery == 0L) run vehicle detector + tracker here.
        return Result(idx, road, ms)
    }
}
