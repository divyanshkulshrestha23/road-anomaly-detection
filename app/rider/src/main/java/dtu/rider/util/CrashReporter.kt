package dtu.rider.util

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Saves any crash's stack trace to Download/RiderAssist/crash_<time>.txt (and the app folder),
 * so testers can send it without a USB cable. Then lets Android show "keeps stopping" as usual.
 */
object CrashReporter {
    fun install(context: Context) {
        val app = context.applicationContext
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            try {
                val sw = StringWriter()
                error.printStackTrace(PrintWriter(sw))
                val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val text = "Rider Assist crash $stamp\nthread=${thread.name}\n" +
                    "device=${Build.MANUFACTURER} ${Build.MODEL} sdk=${Build.VERSION.SDK_INT}\n\n$sw"
                File(app.getExternalFilesDir(null), "crash_$stamp.txt").writeText(text)
                PublicStorage.saveText(app, "crash_$stamp", text)
            } catch (t: Throwable) {
                Log.e("CrashReporter", "Could not save crash report", t)
            }
            previous?.uncaughtException(thread, error)
        }
    }
}
