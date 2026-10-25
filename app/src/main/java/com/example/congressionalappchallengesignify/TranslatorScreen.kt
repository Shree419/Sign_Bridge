package com.example.congressionalappchallengesignify

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

@Composable
fun TranslatorScreen(
    typed: String,
    onTypedChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val activity = context as ComponentActivity

    // The camera callback lives a long time, so it must read the newest values.
    val currentTyped by rememberUpdatedState(typed)
    val currentOnTypedChange by rememberUpdatedState(onTypedChange)

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }
    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    var liveLabel by remember { mutableStateOf<String?>(null) }
    var liveConfidence by remember { mutableStateOf(0f) }
    var handVisible by remember { mutableStateOf(false) }
    var useFront by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    val previewView = remember { PreviewView(context) }

    DisposableEffect(hasPermission, useFront) {
        if (!hasPermission) return@DisposableEffect onDispose { }

        val executor = Executors.newSingleThreadExecutor()
        var analyzer: HandSignAnalyzer? = null
        var provider: ProcessCameraProvider? = null
        try {
            analyzer = HandSignAnalyzer(
                context = context,
                onUpdate = { label, confidence, visible ->
                    liveLabel = label; liveConfidence = confidence; handVisible = visible
                },
                onCommit = { label ->
                    currentOnTypedChange(
                        when (label) {
                            "del" -> currentTyped.dropLast(1)
                            "space" -> currentTyped + " "
                            else -> currentTyped + label
                        }
                    )
                },
            )
        } catch (e: Exception) {
            errorMessage = "Could not load models: ${e.message}"
        }

        val activeAnalyzer = analyzer
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (activeAnalyzer == null) return@addListener
            val cameraProvider = future.get()
            provider = cameraProvider
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(executor, activeAnalyzer) }
            val selector = if (useFront) CameraSelector.DEFAULT_FRONT_CAMERA
            else CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(activity, selector, preview, analysis)
            } catch (e: Exception) {
                errorMessage = "Camera failed: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            provider?.unbindAll()
            executor.shutdown()
            activeAnalyzer?.close()
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        if (hasPermission) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        } else {
            Text(
                "Camera permission is needed to read signs.",
                modifier = Modifier.align(Alignment.Center).padding(24.dp)
            )
        }

        Surface(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(12.dp),
            shape = MaterialTheme.shapes.large,
            tonalElevation = 6.dp,
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                errorMessage?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                Text(
                    when {
                        !handVisible -> "Show your hand to the camera"
                        liveLabel == null -> "Not sure - hold the sign steady"
                        else -> "Seeing: $liveLabel (${(liveConfidence * 100).toInt()}%)"
                    },
                    style = MaterialTheme.typography.bodyMedium
                )
                Text(
                    if (typed.isEmpty()) "Your text will appear here" else typed,
                    style = MaterialTheme.typography.headlineSmall
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { onTypedChange("") }) { Text("Clear") }
                    OutlinedButton(onClick = { onTypedChange(typed.dropLast(1)) }) { Text("Backspace") }
                    OutlinedButton(onClick = { useFront = !useFront }) { Text("Flip") }
                }
            }
        }
    }
}