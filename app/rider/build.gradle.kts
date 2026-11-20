plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "dtu.rider"
    compileSdk = 34

    defaultConfig {
        applicationId = "dtu.rider"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { viewBinding = true }
    // Keep .tflite uncompressed so it can be memory-mapped.
    androidResources { noCompress += "tflite" }
}

dependencies {
    // TensorFlow Lite is now LiteRT (same org.tensorflow.lite Interpreter API, new package name).
    // 1.4.x is the Interpreter line with GPU delegate support. If the model fails to load with an
    // unsupported-op error, try litert:2.3.0 (Interpreter is CPU-only there) and drop litert-gpu.
    val camerax = "1.3.4"
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
}
