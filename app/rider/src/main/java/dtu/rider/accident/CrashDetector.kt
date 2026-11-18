package dtu.rider.accident

import dtu.rider.sensors.SensorLogger
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Multi-signal crash rule (docs/decisions.md #3): an impact alone never triggers an alert.
 *
 *   IDLE --(|a| > impactG)--> IMPACT
 *   IMPACT: within [evalWindowMs], require ALL of
 *       tilt from upright > [tiltDeg] for at least [tiltHoldMs]
 *       GPS speed dropped: was >= [minSpeedBeforeMps], now <= [maxSpeedAfterMps]
 *       stillness: accel magnitude std-dev < [stillStd] over [stillMs]
 *   -> CONFIRMING (countdown [confirmMs], rider can cancel) -> ALERT
 *
 * Thresholds are provisional and must be tuned on recorded rides (deliberate pothole hits,
 * hard braking, and safe simulated falls). Without a GPS speed the rule cannot fire.
 */
class CrashDetector(
    private val logger: SensorLogger,
    private val listener: (State, String) -> Unit,
    private val impactG: Float = 2.0f,
    private val tiltDeg: Float = 50f,
    private val tiltHoldMs: Long = 1000L,
    private val minSpeedBeforeMps: Float = 4.2f,   // ~15 km/h
    private val maxSpeedAfterMps: Float = 1.5f,
    private val stillStd: Float = 0.6f,            // m/s^2
    private val stillMs: Long = 3000L,
    private val evalWindowMs: Long = 10_000L,
    private val confirmMs: Long = 20_000L,
) : SensorLogger.Listener {

    enum class State { IDLE, IMPACT, CONFIRMING, ALERT }

    var state = State.IDLE; private set
    private val g = FloatArray(3)
    private var gInit = false
    private var upright: FloatArray? = null
    private var calibSamples = 0
    private val calibSum = FloatArray(3)

    private var impactAtMs = 0L
    private var speedBefore = 0f
    private var tiltSinceMs = -1L
    private var tiltOk = false
    private var confirmStartMs = 0L

    private val recentSpeeds = ArrayDeque<Pair<Long, Float>>()
    private val magWindow = ArrayDeque<Pair<Long, Float>>()

    /** Call while the bike is upright and steady (e.g. at the start of a ride) to set the reference. */
    fun recalibrate() { upright = null; calibSamples = 0; calibSum.fill(0f) }

    fun cancel() { if (state == State.CONFIRMING) { state = State.IDLE; listener(state, "Cancelled by rider") } }

    override fun onGps(tMs: Long, lat: Double, lon: Double, speedMps: Float?, accuracyM: Float?) {
        if (speedMps == null) return
        val now = System.currentTimeMillis()
        recentSpeeds.addLast(now to speedMps)
        while (recentSpeeds.isNotEmpty() && now - recentSpeeds.first().first > 15_000) recentSpeeds.removeFirst()
    }

    override fun onImu(tNs: Long, ax: Float, ay: Float, az: Float, gx: Float, gy: Float, gz: Float) {
        val now = System.currentTimeMillis()
        // Low-pass gravity estimate.
        if (!gInit) { g[0] = ax; g[1] = ay; g[2] = az; gInit = true }
        val a = 0.9f
        g[0] = a * g[0] + (1 - a) * ax; g[1] = a * g[1] + (1 - a) * ay; g[2] = a * g[2] + (1 - a) * az

        if (upright == null) {
            calibSum[0] += ax; calibSum[1] += ay; calibSum[2] += az
            if (++calibSamples >= 150) upright = floatArrayOf(calibSum[0] / calibSamples, calibSum[1] / calibSamples, calibSum[2] / calibSamples)
            return
        }

        val mag = sqrt(ax * ax + ay * ay + az * az)
        magWindow.addLast(now to mag)
        while (magWindow.isNotEmpty() && now - magWindow.first().first > stillMs) magWindow.removeFirst()

        when (state) {
            State.IDLE -> if (mag / GRAVITY > impactG) {
                state = State.IMPACT; impactAtMs = now; tiltSinceMs = -1L; tiltOk = false
                speedBefore = recentSpeeds.filter { now - it.first <= 5_000 }.maxOfOrNull { it.second } ?: 0f
                listener(state, "Impact %.1f g".format(mag / GRAVITY))
            }
            State.IMPACT -> {
                val tilt = tiltDegrees()
                if (tilt > tiltDeg) {
                    if (tiltSinceMs < 0) tiltSinceMs = now
                    if (now - tiltSinceMs >= tiltHoldMs) tiltOk = true
                } else tiltSinceMs = -1L
                val speedNow = recentSpeeds.lastOrNull()?.second
                val speedOk = speedBefore >= minSpeedBeforeMps && speedNow != null && speedNow <= maxSpeedAfterMps
                val stillOk = now - impactAtMs >= stillMs && std(magWindow.map { it.second }) < stillStd
                if (tiltOk && speedOk && stillOk) {
                    state = State.CONFIRMING; confirmStartMs = now
                    listener(state, "Crash suspected. Tap to cancel (${confirmMs / 1000}s)")
                } else if (now - impactAtMs > evalWindowMs) {
                    state = State.IDLE
                    listener(state, "Impact without crash signs (tilt=$tiltOk speed=$speedOk still=$stillOk)")
                }
            }
            State.CONFIRMING -> if (now - confirmStartMs >= confirmMs) {
                state = State.ALERT
                listener(state, "Emergency alert")   // TODO(Module 4): send SMS with logger.lastLocation
            }
            State.ALERT -> Unit
        }
    }

    private fun tiltDegrees(): Float {
        val u = upright ?: return 0f
        val dot = g[0] * u[0] + g[1] * u[1] + g[2] * u[2]
        val n = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) * sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        if (n == 0f) return 0f
        return Math.toDegrees(acos((dot / n).coerceIn(-1f, 1f)).toDouble()).toFloat()
    }

    private fun std(xs: List<Float>): Float {
        if (xs.size < 2) return Float.MAX_VALUE
        val m = xs.average().toFloat()
        return sqrt(xs.sumOf { ((it - m) * (it - m)).toDouble() }.toFloat() / (xs.size - 1))
    }

    companion object { const val GRAVITY = 9.81f }
}
