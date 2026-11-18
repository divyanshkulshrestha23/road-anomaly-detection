package dtu.rider.detect

import android.graphics.RectF

/** One detection in source-image pixel coordinates. */
data class Detection(
    val classId: Int,
    val label: String,
    val score: Float,
    val box: RectF,
)
