package dtu.rider.detect

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * Runs a YOLO model exported to TFLite by Ultralytics.
 *
 * Expects a float32 input in 0..1, either channels-first [1, 3, H, W] (current Ultralytics LiteRT
 * export) or channels-last [1, H, W, 3] (older TFLite exports); the layout is read from the model.
 * Output is [1, 4 + numClasses, N] or [1, N, 4 + numClasses], with boxes as centre-x, centre-y,
 * width, height, normalised (0..1) or in input pixels; both are handled.
 * detector/export_tflite.py writes model_info.json with the shapes of each export.
 */
class YoloTfliteDetector(
    context: Context,
    modelAsset: String = "detector.tflite",
    labelsAsset: String = "labels.txt",
    private val scoreThreshold: Float = 0.35f,
    private val iouThreshold: Float = 0.5f,
    useGpu: Boolean = true,
) : AutoCloseable {

    val labels: List<String> = context.assets.open(labelsAsset).bufferedReader()
        .readLines().map { it.trim() }.filter { it.isNotEmpty() }

    private var gpuDelegate: GpuDelegate? = null
    private val interpreter: Interpreter
    val inputW: Int
    val inputH: Int
    private val channelsFirstInput: Boolean
    private val inputBuffer: ByteBuffer
    private val inputFloats: FloatArray
    private val outShape: IntArray
    private val output: Array<Array<FloatArray>>
    var usingGpu = false
        private set

    init {
        val model = loadModel(context, modelAsset)
        val opts = Interpreter.Options().apply { setNumThreads(4) }
        if (useGpu) {
            try {
                gpuDelegate = GpuDelegate()
                opts.addDelegate(gpuDelegate)
                usingGpu = true
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate unavailable, using CPU", t)
                gpuDelegate = null
            }
        }
        interpreter = try {
            Interpreter(model, opts)
        } catch (t: Throwable) {
            // Some GPUs reject some ops at load time: retry on CPU.
            Log.w(TAG, "Interpreter failed with GPU delegate, retrying on CPU", t)
            gpuDelegate?.close(); gpuDelegate = null; usingGpu = false
            Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })
        }

        val inT = interpreter.getInputTensor(0)
        require(inT.dataType() == DataType.FLOAT32) {
            "Expected float32 input; use the float export (full-INT8 models need quantised input handling)"
        }
        val inShape = inT.shape()
        channelsFirstInput = inShape[1] == 3
        if (channelsFirstInput) { inputH = inShape[2]; inputW = inShape[3] }   // [1, 3, H, W]
        else { inputH = inShape[1]; inputW = inShape[2] }                      // [1, H, W, 3]
        inputBuffer = ByteBuffer.allocateDirect(4 * inputW * inputH * 3).order(ByteOrder.nativeOrder())
        inputFloats = FloatArray(3 * inputW * inputH)

        outShape = interpreter.getOutputTensor(0).shape()
        output = Array(outShape[0]) { Array(outShape[1]) { FloatArray(outShape[2]) } }
        Log.i(TAG, "Model in=${inShape.contentToString()} (${if (channelsFirstInput) "NCHW" else "NHWC"}) out=${outShape.contentToString()} gpu=$usingGpu labels=${labels.size}")
    }

    private var pixelBuf = IntArray(0)

    /** Detects objects in [bitmap]; returns boxes in the bitmap's pixel coordinates. */
    fun detect(bitmap: Bitmap): List<Detection> {
        // Letterbox to the model input size, keeping aspect ratio.
        val scale = min(inputW.toFloat() / bitmap.width, inputH.toFloat() / bitmap.height)
        val nw = (bitmap.width * scale).toInt()
        val nh = (bitmap.height * scale).toInt()
        val padX = (inputW - nw) / 2f
        val padY = (inputH - nh) / 2f
        val boxed = Bitmap.createBitmap(inputW, inputH, Bitmap.Config.ARGB_8888)
        Canvas(boxed).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(Bitmap.createScaledBitmap(bitmap, nw, nh, true), padX, padY, null)
        }

        val plane = inputW * inputH
        if (pixelBuf.size != plane) pixelBuf = IntArray(plane)
        boxed.getPixels(pixelBuf, 0, inputW, 0, 0, inputW, inputH)
        val f = inputFloats
        if (channelsFirstInput) {
            // Planar RGB: all R, then all G, then all B.
            for (i in 0 until plane) {
                val p = pixelBuf[i]
                f[i] = ((p shr 16) and 0xFF) / 255f
                f[plane + i] = ((p shr 8) and 0xFF) / 255f
                f[2 * plane + i] = (p and 0xFF) / 255f
            }
        } else {
            // Interleaved RGB per pixel.
            for (i in 0 until plane) {
                val p = pixelBuf[i]
                f[3 * i] = ((p shr 16) and 0xFF) / 255f
                f[3 * i + 1] = ((p shr 8) and 0xFF) / 255f
                f[3 * i + 2] = (p and 0xFF) / 255f
            }
        }
        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(f)
        inputBuffer.rewind()
        interpreter.run(inputBuffer, output)

        // Work out layout: channels-first [1, 4+nc, N] or channels-last [1, N, 4+nc].
        val nc = labels.size
        val channelsFirst = outShape[1] == 4 + nc || outShape[1] < outShape[2]
        val n = if (channelsFirst) outShape[2] else outShape[1]
        val o = output[0]
        fun v(c: Int, i: Int) = if (channelsFirst) o[c][i] else o[i][c]

        // Detect whether boxes are normalised.
        var maxCoord = 0f
        for (i in 0 until min(n, 200)) maxCoord = max(maxCoord, max(v(0, i), v(1, i)))
        val sx = if (maxCoord <= 2f) inputW.toFloat() else 1f
        val sy = if (maxCoord <= 2f) inputH.toFloat() else 1f

        val raw = ArrayList<Detection>()
        for (i in 0 until n) {
            var best = -1
            var bestScore = scoreThreshold
            for (c in 0 until nc) {
                val s = v(4 + c, i)
                if (s > bestScore) { bestScore = s; best = c }
            }
            if (best < 0) continue
            val cx = v(0, i) * sx; val cy = v(1, i) * sy
            val w = v(2, i) * sx; val h = v(3, i) * sy
            // Undo letterbox -> source bitmap pixels.
            val l = ((cx - w / 2) - padX) / scale
            val t = ((cy - h / 2) - padY) / scale
            val r = ((cx + w / 2) - padX) / scale
            val b = ((cy + h / 2) - padY) / scale
            raw.add(Detection(best, labels.getOrElse(best) { "class_$best" }, bestScore,
                RectF(max(0f, l), max(0f, t), min(bitmap.width.toFloat(), r), min(bitmap.height.toFloat(), b))))
        }
        return nms(raw)
    }

    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }.toMutableList()
        val keep = ArrayList<Detection>()
        while (sorted.isNotEmpty()) {
            val d = sorted.removeAt(0)
            keep.add(d)
            sorted.removeAll { it.classId == d.classId && iou(it.box, d.box) > iouThreshold }
        }
        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val ix = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val iy = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val inter = ix * iy
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    companion object {
        private const val TAG = "YoloDetector"

        private fun loadModel(context: Context, asset: String): MappedByteBuffer {
            context.assets.openFd(asset).use { fd ->
                FileInputStream(fd.fileDescriptor).use { fis ->
                    return fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
        }
    }
}
