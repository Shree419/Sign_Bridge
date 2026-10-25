package com.example.congressionalappchallengesignify

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

// Put your name or team name here.
private const val CREATOR = "Your name here"

@Composable
fun AboutScreen(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            "About $APP_NAME",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        AboutCard(
            "Why we built it",
            "Many Deaf and hard-of-hearing people sign, but many hearing people don't know ASL. " +
                    "$APP_NAME is a small step toward helping the two groups understand each other " +
                    "by turning fingerspelled letters into text.",
        )
        AboutCard(
            "How it works",
            "The camera finds your hand and tracks 21 points on it using Google's MediaPipe. " +
                    "A small neural network, trained on the public Kaggle ASL Alphabet dataset, " +
                    "reads those points and picks the most likely letter. All of this runs on your phone.",
        )
        AboutCard(
            "Privacy",
            "Camera frames are processed on the device and are never saved or uploaded.",
        )
        AboutCard(
            "What it can't do (yet)",
            "ASL is a full language with its own grammar, and this app reads only the fingerspelled " +
                    "alphabet, not whole signs or sentences. Letters that involve motion, like J and Z, " +
                    "are hard to recognize from a single frame, and closed-fist letters like M and N can " +
                    "be less reliable. Results also vary with lighting and hand position.",
        )
        AboutCard(
            "Tips for best results",
            "Use good lighting, keep your whole hand in view, hold each letter steady for about a " +
                    "second, and lower your hand before repeating the same letter.",
        )
        AboutCard(
            "Credits",
            "Created by $CREATOR for the Congressional App Challenge. Hand tracking: Google MediaPipe. " +
                    "Training data: Kaggle ASL Alphabet dataset.",
        )
    }
}

@Composable
private fun AboutCard(title: String, body: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(body, style = MaterialTheme.typography.bodyMedium)
        }
    }
}