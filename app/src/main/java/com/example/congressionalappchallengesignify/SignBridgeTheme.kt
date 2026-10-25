package com.example.congressionalappchallengesignify

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

object Brand {
    val Cream = Color(0xFFFAF3E7)
    val Sand = Color(0xFFEADFCB)
    val Sage = Color(0xFFCFE3D6)
    val Teal = Color(0xFF0F5A56)
    val TealLight = Color(0xFF1C7A74)
    val Coral = Color(0xFFE8604C)
    val Sun = Color(0xFFF4B942)
    val Ink = Color(0xFF1F2A28)
}

private val Heading = FontFamily.Serif
private val Body = FontFamily.SansSerif

private val SignTypography = Typography(
    displaySmall = TextStyle(
        fontFamily = Heading, fontWeight = FontWeight.Bold,
        fontSize = 44.sp, lineHeight = 48.sp, letterSpacing = (-0.5).sp,
    ),
    headlineMedium = TextStyle(
        fontFamily = Heading, fontWeight = FontWeight.Bold,
        fontSize = 32.sp, lineHeight = 38.sp,
    ),
    headlineSmall = TextStyle(
        fontFamily = Heading, fontWeight = FontWeight.SemiBold,
        fontSize = 26.sp, lineHeight = 32.sp,
    ),
    titleLarge = TextStyle(
        fontFamily = Heading, fontWeight = FontWeight.SemiBold,
        fontSize = 22.sp, lineHeight = 28.sp,
    ),
    titleMedium = TextStyle(
        fontFamily = Heading, fontWeight = FontWeight.SemiBold,
        fontSize = 18.sp, lineHeight = 24.sp,
    ),
    bodyLarge = TextStyle(
        fontFamily = Body, fontWeight = FontWeight.Normal,
        fontSize = 17.sp, lineHeight = 26.sp,
    ),
    bodyMedium = TextStyle(
        fontFamily = Body, fontWeight = FontWeight.Normal,
        fontSize = 15.sp, lineHeight = 23.sp,
    ),
    labelLarge = TextStyle(
        fontFamily = Body, fontWeight = FontWeight.SemiBold,
        fontSize = 15.sp, lineHeight = 20.sp,
    ),
)

private val SignColors = lightColorScheme(
    primary = Brand.Teal,
    onPrimary = Brand.Cream,
    secondary = Brand.Coral,
    onSecondary = Color.White,
    tertiary = Brand.Sun,
    background = Brand.Cream,
    onBackground = Brand.Ink,
    surface = Brand.Cream,
    onSurface = Brand.Ink,
    surfaceVariant = Brand.Sand,
    onSurfaceVariant = Brand.Ink,
)

@Composable
fun SignBridgeTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = SignColors,
        typography = SignTypography,
        content = content,
    )
}