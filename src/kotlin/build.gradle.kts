plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.opencv_tutorial"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.opencv_tutorial"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // Add NDK ABI filters to ensure compatibility
        ndk {
            abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64"))
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isDebuggable = true
            // Enable more detailed native logging for debugging
            buildConfigField("boolean", "ENABLE_DETAILED_LOGGING", "true")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.1"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            // Avoid duplicate library files
            pickFirst("**/libc++_shared.so")
            pickFirst("**/libOpenCL.so")
        }
        jniLibs {
            useLegacyPackaging = true // Helps with native lib compatibility
        }
    }

    // Add for better compatibility with native libraries
    ndkVersion = "21.4.7075529" // Use a stable NDK version
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(project(":sdk"))
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // Replace with more specific version
    implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")

    // Other dependencies...
    implementation("androidx.compose.ui:ui:1.5.1")
    implementation("androidx.compose.material:material:1.5.1")
    implementation("androidx.compose.ui:ui-tooling-preview:1.5.1")
    implementation("androidx.activity:activity-compose:1.7.2")
    debugImplementation("androidx.compose.ui:ui-tooling:1.5.1")
    implementation("org.tensorflow:tensorflow-lite:2.9.0")
    implementation("org.tensorflow:tensorflow-lite-task-vision:0.4.2")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.9.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.2")
    
    // Add metadata extractor for better model information
    implementation("org.tensorflow:tensorflow-lite-metadata:0.4.2")
}