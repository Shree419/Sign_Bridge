package com.example.congressionalappchallengesignify

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun AboutScreen(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Brand.Cream)
            .verticalScroll(rememberScrollState()),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(bottomStart = 44.dp, bottomEnd = 44.dp))
                .background(Brand.Teal)
                .statusBarsPadding()
                .padding(horizontal = 28.dp)
                .padding(top = 36.dp, bottom = 40.dp),
        ) {
            Text(
                "ABOUT",
                style = MaterialTheme.typography.labelLarge,
                color = Brand.Sun,
                letterSpacing = 2.sp,
            )
            Spacer(Modifier.height(8.dp))
            Text(APP_NAME, style = MaterialTheme.typography.displaySmall, color = Brand.Cream)
        }

        Column(
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 28.dp),
            verticalArrangement = Arrangement.spacedBy(22.dp),
        ) {
            Section(
                "Why we made this",
                "Most people don't know any ASL, so a Deaf or hard-of-hearing person can end up " +
                        "spelling something out with nobody able to read it. We wanted something simple " +
                        "that turns those letters into text right away.",
            )
            Divider()
            Section(
                "How it works",
                "The camera finds your hand and marks 21 points on it, like each knuckle and " +
                        "fingertip. A small model we trained on a public set of ASL alphabet photos looks " +
                        "at those points and guesses the letter. Hold a sign steady and it gets typed.",
            )
            Divider()
            Section(
                "Privacy",
                "Video is processed on your phone while the camera is open. It isn't saved or sent anywhere.",
            )
            Divider()
            Section(
                "What it can't do yet",
                "It reads the fingerspelled alphabet only, not full ASL signs or sentences. " +
                        "J and Z use movement, so they are harder to catch, and M and N are less " +
                        "reliable too. Bad lighting or a partly hidden hand will cause mistakes.",
            )
            Divider()
            Section(
                "Tips",
                "Use good light, keep your whole hand in view, and hold each letter for about a " +
                        "second. Lower your hand before repeating a letter, like the two L's in HELLO.",
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(28.dp))
                    .background(Brand.Sage)
                    .padding(24.dp),
            ) {
                Text(
                    "BUILT BY",
                    style = MaterialTheme.typography.labelLarge,
                    color = Brand.Teal,
                    letterSpacing = 2.sp,
                )
                Spacer(Modifier.height(10.dp))
                Text(
                    "Shreyansh Guduru",
                    style = MaterialTheme.typography.headlineSmall,
                    color = Brand.Ink,
                )
                Text(
                    "Rajan Sankara Subramanian",
                    style = MaterialTheme.typography.headlineSmall,
                    color = Brand.Ink,
                )
                Spacer(Modifier.height(14.dp))
                Text(
                    "Made for the Congressional App Challenge. Hand tracking by Google MediaPipe. " +
                            "Letters learned from the Kaggle ASL Alphabet dataset.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Brand.Ink.copy(alpha = 0.8f),
                )
            }
        }
    }
}

@Composable
private fun Section(title: String, body: String) {
    Column {
        Text(title, style = MaterialTheme.typography.titleLarge, color = Brand.Teal)
        Spacer(Modifier.height(6.dp))
        Text(
            body,
            style = MaterialTheme.typography.bodyLarge,
            color = Brand.Ink.copy(alpha = 0.85f),
        )
    }
}

@Composable
private fun Divider() {
    Box(
        Modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(Brand.Sand)
    )
}