package dtu.rider.emergency

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.os.Build
import android.telephony.SmsManager
import android.util.Log
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Stores the emergency contact and sends the crash SMS with a map link. */
class EmergencySms(private val context: Context) {
    private val prefs = context.getSharedPreferences("rider_assist", Context.MODE_PRIVATE)

    var contact: String?
        get() = prefs.getString("emergency_contact", null)?.takeIf { it.isNotBlank() }
        set(v) { prefs.edit().putString("emergency_contact", v?.trim()).apply() }

    fun hasPermission() =
        ContextCompat.checkSelfPermission(context, Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED

    fun message(location: Location?, test: Boolean): String {
        val time = SimpleDateFormat("HH:mm", Locale.US).format(Date())
        val where = location?.let {
            "Location: https://maps.google.com/?q=%.6f,%.6f (±%.0f m)".format(Locale.US, it.latitude, it.longitude, if (it.hasAccuracy()) it.accuracy else 0f)
        } ?: "Location unavailable (no GPS fix)."
        val head = if (test) "TEST from Rider Assist, please ignore." else "Rider Assist: possible crash detected at $time. The rider did not cancel the alert."
        return "$head $where"
    }

    /** Returns null on success, or an error message. */
    fun send(location: Location?, test: Boolean = false): String? {
        val to = contact ?: return "No emergency contact set"
        if (!hasPermission()) return "SMS permission not granted"
        return try {
            val sms: SmsManager = if (Build.VERSION.SDK_INT >= 31) context.getSystemService(SmsManager::class.java)
            else @Suppress("DEPRECATION") SmsManager.getDefault()
            val parts = sms.divideMessage(message(location, test))
            sms.sendMultipartTextMessage(to, null, parts, null, null)
            Log.i("EmergencySms", "SMS sent to $to (test=$test)")
            null
        } catch (t: Throwable) {
            Log.e("EmergencySms", "SMS failed", t)
            "SMS failed: ${t.message}"
        }
    }
}
