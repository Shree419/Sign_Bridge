package com.example.congressionalappchallengesignify

import android.content.Context
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

/** Runs asl_landmarks.tflite on the 21 hand landmarks from MediaPipe. */
class AslClassifier(context: Context) {
    private val interpreter: Interpreter
    private val labels: List<String> =
        context.assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }

    init {
        val fd = context.assets.openFd("asl_landmarks.tflite")
        val buffer = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(buffer)
    }

    data class Result(val label: String, val confidence: Float)

    /** Landmarks go in RAW; normalisation happens inside the model. Null = not confident. */
    fun classify(landmarks: List<NormalizedLandmark>, minConfidence: Float = 0.6f): Result? {
        if (landmarks.size != 21) return null
        val input = Array(1) { FloatArray(63) }
        landmarks.forEachIndexed { i, p ->
            input[0][i * 3] = p.x()
            input[0][i * 3 + 1] = p.y()
            input[0][i * 3 + 2] = p.z()
        }
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)
        val best = output[0].indices.maxByOrNull { output[0][it] } ?: return null
        val conf = output[0][best]
        return if (conf >= minConfidence) Result(labels[best], conf) else null
    }

    fun close() = interpreter.close()
}