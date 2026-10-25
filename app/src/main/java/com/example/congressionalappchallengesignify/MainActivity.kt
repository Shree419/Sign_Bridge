package com.example.congressionalappchallengesignify

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import com.example.congressionalappchallengesignify.ui.theme.CongressionalAppChallengeSignifyTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CongressionalAppChallengeSignifyTheme {
                AppRoot()
            }
        }
    }
}

enum class Page(val label: String, val icon: ImageVector) {
    HOME("Home", Icons.Filled.Home),
    TRANSLATOR("Translator", Icons.Filled.Face),
    ABOUT("About", Icons.Filled.Info),
}

@Composable
fun AppRoot() {
    var page by rememberSaveable { mutableStateOf(Page.HOME) }
    // Kept here so the text survives switching tabs.
    var typed by rememberSaveable { mutableStateOf("") }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        bottomBar = {
            NavigationBar {
                Page.entries.forEach { p ->
                    NavigationBarItem(
                        selected = p == page,
                        onClick = { page = p },
                        icon = { Icon(p.icon, contentDescription = p.label) },
                        label = { Text(p.label) },
                    )
                }
            }
        }
    ) { innerPadding ->
        when (page) {
            Page.HOME -> HomeScreen(
                onStart = { page = Page.TRANSLATOR },
                modifier = Modifier.padding(innerPadding),
            )
            Page.TRANSLATOR -> TranslatorScreen(
                typed = typed,
                onTypedChange = { typed = it },
                modifier = Modifier.padding(innerPadding),
            )
            Page.ABOUT -> AboutScreen(modifier = Modifier.padding(innerPadding))
        }
    }
}