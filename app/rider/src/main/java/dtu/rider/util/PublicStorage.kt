package dtu.rider.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.OutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Copies app output to places the user can see in Files / Gallery:
 *   Snap photos  -> Pictures/RiderAssist/
 *   Recordings   -> Download/RiderAssist/ride_<time>.zip
 * Android 10+ (API 29+): via MediaStore, no permission needed.
 * Android 8-9: plain files, needs WRITE_EXTERNAL_STORAGE (declared with maxSdkVersion 28).
 */
object PublicStorage {
    private const val FOLDER = "RiderAssist"

    /** Returns a human-readable location, or throws on failure. */
    fun saveImage(context: Context, bitmap: Bitmap, name: String): String {
        val file = "$name.jpg"
        write(context, file, "image/jpeg", Environment.DIRECTORY_PICTURES,
            if (Build.VERSION.SDK_INT >= 29) MediaStore.Images.Media.EXTERNAL_CONTENT_URI else null) { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
        return "Pictures/$FOLDER/$file"
    }

    /** Writes a text file to Download/RiderAssist/<name>.txt. */
    fun saveText(context: Context, name: String, text: String): String {
        val file = "$name.txt"
        write(context, file, "text/plain", Environment.DIRECTORY_DOWNLOADS,
            if (Build.VERSION.SDK_INT >= 29) MediaStore.Downloads.EXTERNAL_CONTENT_URI else null) { out ->
            out.write(text.toByteArray())
        }
        return "Download/$FOLDER/$file"
    }

    /** Zips every file in [dir] into Download/RiderAssist/<dir name>.zip. */
    fun exportZip(context: Context, dir: File): String {
        val file = "${dir.name}.zip"
        write(context, file, "application/zip", Environment.DIRECTORY_DOWNLOADS,
            if (Build.VERSION.SDK_INT >= 29) MediaStore.Downloads.EXTERNAL_CONTENT_URI else null) { out ->
            ZipOutputStream(out).use { zip ->
                dir.listFiles()?.filter { it.isFile }?.forEach { f ->
                    zip.putNextEntry(ZipEntry("${dir.name}/${f.name}"))
                    f.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
            }
        }
        return "Download/$FOLDER/$file"
    }

    private fun write(context: Context, name: String, mime: String, publicDir: String, collection: Uri?, body: (OutputStream) -> Unit) {
        if (Build.VERSION.SDK_INT >= 29 && collection != null) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(MediaStore.MediaColumns.RELATIVE_PATH, "$publicDir/$FOLDER")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val resolver = context.contentResolver
            val uri = resolver.insert(collection, values) ?: error("Could not create $name")
            try {
                resolver.openOutputStream(uri)?.use(body) ?: error("Could not open $name")
                values.clear(); values.put(MediaStore.MediaColumns.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            } catch (t: Throwable) {
                resolver.delete(uri, null, null); throw t
            }
        } else {
            @Suppress("DEPRECATION")
            val dir = File(Environment.getExternalStoragePublicDirectory(publicDir), FOLDER).apply { mkdirs() }
            val f = File(dir, name)
            f.outputStream().use(body)
            MediaScannerConnection.scanFile(context, arrayOf(f.absolutePath), arrayOf(mime), null)
        }
    }
}
