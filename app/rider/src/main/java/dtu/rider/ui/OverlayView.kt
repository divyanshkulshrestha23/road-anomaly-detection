package dtu.rider.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import dtu.rider.detect.Detection
import kotlin.math.min

/** Draws detections on top of a PreviewView that uses scaleType="fitCenter". */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var imgW = 1
    private var imgH = 1
    private var pathZone: RectF? = null

    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 5f; isAntiAlias = true }
    private val textPaint = Paint().apply { color = Color.WHITE; textSize = 34f; isAntiAlias = true }
    private val bgPaint = Paint().apply { style = Paint.Style.FILL }
    private val zonePaint = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 3f; color = Color.argb(140, 255, 255, 0)
        pathEffect = android.graphics.DashPathEffect(floatArrayOf(20f, 12f), 0f)
    }

    fun update(dets: List<Detection>, imageWidth: Int, imageHeight: Int, zone: RectF?) {
        detections = dets; imgW = imageWidth; imgH = imageHeight; pathZone = zone
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val s = min(width.toFloat() / imgW, height.toFloat() / imgH)
        val ox = (width - imgW * s) / 2f
        val oy = (height - imgH * s) / 2f
        fun map(r: RectF) = RectF(ox + r.left * s, oy + r.top * s, ox + r.right * s, oy + r.bottom * s)

        pathZone?.let { canvas.drawRect(map(it), zonePaint) }
        for (d in detections) {
            val c = colorFor(d.classId)
            boxPaint.color = c; bgPaint.color = c
            val r = map(d.box)
            canvas.drawRect(r, boxPaint)
            // Tracked vehicles (classId >= 100) already carry distance/TTC in the label.
            val text = if (d.classId >= 100) d.label else "${d.label} ${(d.score * 100).toInt()}%"
            val tw = textPaint.measureText(text)
            canvas.drawRect(r.left, r.top - 40f, r.left + tw + 12f, r.top, bgPaint)
            canvas.drawText(text, r.left + 6f, r.top - 10f, textPaint)
        }
    }

    private fun colorFor(id: Int) = when (id) {
        0 -> Color.rgb(229, 57, 53)    // pothole
        1 -> Color.rgb(251, 140, 0)
        2 -> Color.rgb(253, 216, 53)
        3 -> Color.rgb(142, 36, 170)
        in 4..99 -> Color.rgb(67, 160, 71)    // road signs
        else -> Color.rgb(30, 136, 229)       // tracked vehicles (100+)
    }
}
