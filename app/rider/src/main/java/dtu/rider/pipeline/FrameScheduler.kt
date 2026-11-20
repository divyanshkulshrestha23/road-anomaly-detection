package dtu.rider.pipeline

import android.graphics.Bitmap
import android.os.SystemClock
import dtu.rider.detect.Detection
import dtu.rider.detect.YoloTfliteDetector

/**
 * Decides which model runs on which frame: the shared road detector (potholes, cracks, signs)
 * every [roadEvery] frames and the vehicle detector (Module 3) every [vehicleEvery] frames,
 * offset so the two rarely land on the same frame.
 */
class FrameScheduler(
    private val roadDetector: YoloTfliteDetector?,
    private val vehicleDetector: YoloTfliteDetector? = null,
    private val roadEvery: Int = 1,
    private val vehicleEvery: Int = 2,
) {
    data class Result(
        val frameIndex: Long,
        val road: List<Detection>?,      // null when the road model was skipped this frame
        val vehicles: List<Detection>?,  // null when the vehicle model was skipped this frame
        val inferenceMs: Long,           // road model time this frame
        val vehicleMs: Long,             // vehicle model time this frame
    )

    private var frame = 0L

    fun process(bitmap: Bitmap): Result {
        val idx = frame++
        var ms = 0L
        var vms = 0L
        var road: List<Detection>? = null
        var vehicles: List<Detection>? = null
        if (roadDetector != null && idx % roadEvery == 0L) {
            val t0 = SystemClock.elapsedRealtime()
            road = roadDetector.detect(bitmap)
            ms = SystemClock.elapsedRealtime() - t0
        }
        if (vehicleDetector != null && (idx + 1) % vehicleEvery == 0L) {
            val t0 = SystemClock.elapsedRealtime()
            vehicles = vehicleDetector.detect(bitmap)
            vms = SystemClock.elapsedRealtime() - t0
        }
        return Result(idx, road, vehicles, ms, vms)
    }
}
