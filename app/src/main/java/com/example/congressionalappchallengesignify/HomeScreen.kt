package com.example.congressionalappchallengesignify

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

// Change this one line to rename the app inside the screens.
const val APP_NAME = "SignBridge"

@Composable
fun HomeScreen(onStart: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            APP_NAME,
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )
        Text(
            "Turn ASL fingerspelling into text, live, on your phone.",
            style = MaterialTheme.typography.titleMedium,
        )

        Button(onClick = onStart, modifier = Modifier.fillMaxWidth()) {
            Text("Start translating")
        }

        HomeFeatureCard(
            "Private by design",
            "Your camera video is processed on your phone. Nothing is uploaded.",
        )
        HomeFeatureCard(
            "Works offline",
            "The recognition models are built into the app, so no internet is needed.",
        )
        HomeFeatureCard(
            "Live results",
            "Hold a letter steady and it appears as text within about a second.",
        )

        Text(
            "How it works",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        HomeStepRow(1, "Open the Translator tab and allow camera access.")
        HomeStepRow(2, "Hold one ASL letter in front of the camera.")
        HomeStepRow(3, "Keep it steady until the letter is added to your text.")
        HomeStepRow(4, "Lower your hand between repeated letters, like the two L's in \"HELLO\".")
    }
}

@Composable
private fun HomeFeatureCard(title: String, body: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(body, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun HomeStepRow(number: Int, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Surface(
            shape = CircleShape,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(32.dp),
        ) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    "$number",
                    color = MaterialTheme.colorScheme.onPrimary,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
        Spacer(Modifier.width(12.dp))
        Text(text, style = MaterialTheme.typography.bodyLarge)
    }
}