package com.example.congressionalappchallengesignify

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// Change this one line to rename the app inside the screens.
const val APP_NAME = "SignBridge"

@Composable
fun HomeScreen(onStart: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Brand.Cream)
            .verticalScroll(rememberScrollState()),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(bottomStart = 44.dp, bottomEnd = 44.dp))
                .background(Brand.Teal),
        ) {
            Box(
                Modifier
                    .align(Alignment.TopEnd)
                    .offset(x = 70.dp, y = (-40).dp)
                    .size(220.dp)
                    .background(Brand.TealLight, CircleShape)
            )
            Box(
                Modifier
                    .align(Alignment.BottomStart)
                    .offset(x = (-50).dp, y = 60.dp)
                    .size(150.dp)
                    .background(Brand.Coral, CircleShape)
            )
            Box(
                Modifier
                    .align(Alignment.CenterEnd)
                    .offset(x = (-34).dp, y = 40.dp)
                    .size(46.dp)
                    .background(Brand.Sun, CircleShape)
            )
            Column(
                modifier = Modifier
                    .statusBarsPadding()
                    .padding(horizontal = 28.dp)
                    .padding(top = 48.dp, bottom = 56.dp),
            ) {
                Text(
                    "ASL FINGERSPELLING",
                    style = MaterialTheme.typography.labelLarge,
                    color = Brand.Sun,
                    letterSpacing = 2.sp,
                )
                Spacer(Modifier.height(12.dp))
                Text(APP_NAME, style = MaterialTheme.typography.displaySmall, color = Brand.Cream)
                Spacer(Modifier.height(12.dp))
                Text(
                    "Spell with your hands.\nRead it on your screen.",
                    style = MaterialTheme.typography.bodyLarge,
                    color = Brand.Sage,
                )
                Spacer(Modifier.height(28.dp))
                Button(
                    onClick = onStart,
                    shape = RoundedCornerShape(50),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Brand.Coral,
                        contentColor = Color.White,
                    ),
                    contentPadding = PaddingValues(horizontal = 28.dp, vertical = 14.dp),
                ) {
                    Text("Start translating", style = MaterialTheme.typography.labelLarge)
                }
            }
        }

        Column(
            modifier = Modifier.padding(horizontal = 28.dp, vertical = 32.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            Point(
                Brand.Coral,
                "Stays on your phone",
                "The camera feed is read right on your device. Nothing is recorded or sent anywhere.",
            )
            Point(
                Brand.Sun,
                "Works without Wi-Fi",
                "Everything the app needs is already inside it.",
            )
            Point(
                Brand.Teal,
                "Good for names and words you spell out",
                "Names, places, anything you would fingerspell one letter at a time.",
            )
            Text(
                "A Congressional App Challenge entry",
                style = MaterialTheme.typography.bodyMedium,
                color = Brand.Ink.copy(alpha = 0.55f),
            )
        }
    }
}

@Composable
private fun Point(dot: Color, title: String, body: String) {
    Row {
        Box(
            Modifier
                .padding(top = 7.dp)
                .size(12.dp)
                .background(dot, CircleShape)
        )
        Spacer(Modifier.width(14.dp))
        Column {
            Text(title, style = MaterialTheme.typography.titleMedium, color = Brand.Ink)
            Spacer(Modifier.height(3.dp))
            Text(
                body,
                style = MaterialTheme.typography.bodyMedium,
                color = Brand.Ink.copy(alpha = 0.75f),
            )
        }
    }
}