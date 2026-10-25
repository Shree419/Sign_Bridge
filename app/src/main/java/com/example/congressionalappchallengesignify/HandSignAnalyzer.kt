package com.example.congressionalappchallengesignify

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

/**
 * Camera frame -> MediaPipe hand landmarks -> ASL classifier -> smoothed letters.
 * A letter is "typed" after it is held for STREAK_NEEDED frames in a row.
 * To type the same letter twice, lower your hand (or change sign) in between.
 */
class HandSignAnalyzer(
    context: Context,
    private val onUpdate: (label: String?, confidence: Float, handVisible: Boolean) -> Unit,
    private val onCommit: (label: String) -> Unit,
) : ImageAnalysis.Analyzer {

    private val classifier = AslClassifier(context)
    private val landmarker: HandLandmarker
    private val main = Handler(Looper.getMainLooper())

    @Volatile private var closed = false
    private var lastTimestamp = 0L
    private var candidate: String? = null
    private var streak = 0
    private var committed = false

    init {
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()
        landmarker = HandLandmarker.createFromOptions(context, options)
    }

    override fun analyze(image: ImageProxy) {
        if (closed) { image.close(); return }
        try {
            val raw = image.toBitmap()
            val rotation = image.imageInfo.rotationDegrees
            val bitmap = if (rotation != 0) {
                val m = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
            } else raw

            var ts = SystemClock.uptimeMillis()
            if (ts <= lastTimestamp) ts = lastTimestamp + 1
            lastTimestamp = ts

            val result = landmarker.detectForVideo(BitmapImageBuilder(bitmap).build(), ts)
            val hand = result.landmarks().firstOrNull()

            if (hand == null) {
                candidate = null; streak = 0; committed = false
                post(null, 0f, false)
                return
            }

            val prediction = classifier.classify(hand)
            if (prediction == null) {
                streak = 0          // low confidence: restart the hold counter
                post(null, 0f, true)
                return
            }

            if (prediction.label == candidate) {
                streak++
            } else {
                candidate = prediction.label; streak = 1; committed = false
            }
            if (streak >= STREAK_NEEDED && !committed) {
                committed = true
                val label = prediction.label
                main.post { onCommit(label) }
            }
            post(prediction.label, prediction.confidence, true)
        } catch (e: Exception) {
            android.util.Log.e("HandSignAnalyzer", "frame failed", e)
        } finally {
            image.close()
        }
    }

    private fun post(label: String?, confidence: Float, handVisible: Boolean) {
        main.post { onUpdate(label, confidence, handVisible) }
    }

    fun close() {
        closed = true
        try { landmarker.close() } catch (_: Exception) {}
        try { classifier.close() } catch (_: Exception) {}
    }

    companion object { private const val STREAK_NEEDED = 10 }
}