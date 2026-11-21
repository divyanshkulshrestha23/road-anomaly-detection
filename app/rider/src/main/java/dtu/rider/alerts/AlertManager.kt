package dtu.rider.alerts

import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.TextView
import java.util.PriorityQueue

/** Priority of an alert; lower ordinal = more urgent. */
enum class AlertLevel { CRASH, COLLISION, POTHOLE, SIGN }

data class Alert(val level: AlertLevel, val message: String, val key: String, val timeMs: Long = System.currentTimeMillis())

/**
 * Shows one alert at a time, most urgent first, and suppresses repeats of the same key
 * within a cooldown so the rider is not flooded.
 */
class AlertManager(private val banner: TextView, private val cooldownMs: Long = 4000L, private val showMs: Long = 2000L) {
    private val main = Handler(Looper.getMainLooper())
    private val queue = PriorityQueue<Alert>(compareBy<Alert>({ it.level.ordinal }, { it.timeMs }))
    private val lastShown = HashMap<String, Long>()
    private var showing = false
    private val tone = try { ToneGenerator(AudioManager.STREAM_NOTIFICATION, 90) } catch (_: Throwable) { null }

    /** Safe to call from any thread: all state is touched on the main thread only. */
    fun post(alert: Alert) {
        main.post {
            val now = System.currentTimeMillis()
            val last = lastShown[alert.key]
            if (last != null && now - last < cooldownMs) return@post
            lastShown[alert.key] = now
            queue.add(alert)
            if (!showing) showNext()
        }
    }

    private fun showNext() {
        val a = queue.poll() ?: run { showing = false; banner.visibility = View.GONE; return }
        showing = true
        banner.text = a.message
        banner.visibility = View.VISIBLE
        if (a.level <= AlertLevel.POTHOLE) tone?.startTone(ToneGenerator.TONE_PROP_BEEP2, 250)
        main.postDelayed({ showNext() }, showMs)
    }

    fun release() { tone?.release() }
}
