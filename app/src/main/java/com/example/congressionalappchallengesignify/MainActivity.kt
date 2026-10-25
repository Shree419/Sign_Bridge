package com.example.congressionalappchallengesignify

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.light(
                android.graphics.Color.TRANSPARENT,
                android.graphics.Color.TRANSPARENT
            ),
        )
        setContent {
            SignBridgeTheme {
                AppRoot()
            }
        }
    }
}

enum class Page(val label: String) {
    HOME("Home"),
    TRANSLATOR("Translate"),
    ABOUT("About"),
}

@Composable
fun AppRoot() {
    var page by rememberSaveable { mutableStateOf(Page.HOME) }
    // Kept here so the text survives switching tabs.
    var typed by rememberSaveable { mutableStateOf("") }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Brand.Cream,
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        bottomBar = { BottomTabs(current = page, onSelect = { page = it }) },
    ) { innerPadding ->
        val content = Modifier.padding(bottom = innerPadding.calculateBottomPadding())
        when (page) {
            Page.HOME -> HomeScreen(onStart = { page = Page.TRANSLATOR }, modifier = content)
            Page.TRANSLATOR -> TranslatorScreen(
                typed = typed,
                onTypedChange = { typed = it },
                modifier = content,
            )
            Page.ABOUT -> AboutScreen(modifier = content)
        }
    }
}

@Composable
private fun BottomTabs(current: Page, onSelect: (Page) -> Unit) {
    Surface(color = Brand.Cream, shadowElevation = 10.dp) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            Page.entries.forEach { p ->
                val selected = p == current
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(50))
                        .background(if (selected) Brand.Teal else Color.Transparent)
                        .clickable { onSelect(p) }
                        .padding(horizontal = 22.dp, vertical = 10.dp),
                ) {
                    Text(
                        p.label,
                        style = MaterialTheme.typography.labelLarge,
                        color = if (selected) Brand.Cream else Brand.Ink.copy(alpha = 0.7f),
                    )
                }
            }
        }
    }
}