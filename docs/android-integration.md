# Android Integration Guide (Kotlin & C++)

This document explains how to embed the converted ONNX ASR model into an Android application using **Kotlin** and **ONNX Runtime Mobile**, with notes on optional **C++ / NDK acceleration**.

---

## 1. Prerequisites for Android
- **Android Studio**: Arctic Fox or newer.
- **Minimum SDK (minSdk)**: API Level 26 (Android 8.0 Oreo) or higher.
- **Target SDK (targetSdk)**: API Level 34+ (Android 14+).
- **Target Architecture**: `arm64-v8a` (recommended for optimal performance), `armeabi-v7a`, `x86_64` (emulators).

---

## 2. Gradle Dependencies

In your application module's `build.gradle.kts`:

```kotlin
dependencies {
    // ONNX Runtime Mobile for Android
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
    
    // Coroutines for background inference
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

Prevent Gradle from compressing the `.onnx` model files in your APK so they can be memory-mapped directly:
```kotlin
android {
    aaptOptions {
        noCompress("onnx")
    }
}
```

---

## 3. Asset Placement

Extract the files from `output/android/ethio-94m-int8/` directly into your Android project's assets folder:

```text
app/src/main/assets/asr/
├── model.onnx                  # 117 MB (INT8 Quantized Model)
├── vocab.json                  # Vocabulary token-to-character mapping
├── manifest.json               # Model metadata & configuration
└── SHA256SUMS                  # Checksum manifest
```

---

## 4. Kotlin Integration Code

We provide a complete, production-ready class: [EthioASRRecognizer.kt](output/android/EthioASRRecognizer.kt).

### Basic Usage Example

```kotlin
package com.example.speechapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.ethioasr.recognizer.EthioASRRecognizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var asrRecognizer: EthioASRRecognizer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize recognizer (loads ONNX model and vocab from assets/asr/)
        asrRecognizer = EthioASRRecognizer(this, "asr/model.onnx")
    }

    private fun processRecordedAudio(pcmFloatSamples: FloatArray) {
        lifecycleScope.launch(Dispatchers.Default) {
            // Run inference on background thread
            val text = asrRecognizer.transcribe(pcmFloatSamples)
            
            withContext(Dispatchers.Main) {
                // Update UI with transcribed Ethiopic/Latin text
                println("Transcription: $text")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        asrRecognizer.close()
    }
}
```

---

## 5. Audio Capture & Preprocessing Contract

The model expects mono 16 kHz audio samples normalized to zero-mean and unit-variance.

If you are recording audio using Android's `AudioRecord`:
```kotlin
val sampleRate = 16000
val channelConfig = AudioFormat.CHANNEL_IN_MONO
val audioFormat = AudioFormat.ENCODING_PCM_16BIT
val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

val audioRecord = AudioRecord(
    MediaRecorder.AudioSource.MIC,
    sampleRate,
    channelConfig,
    audioFormat,
    minBufferSize
)
```

Convert short PCM data to `FloatArray` (-1.0f to 1.0f):
```kotlin
fun shortToFloat(shortBuffer: ShortArray): FloatArray {
    val floatArray = FloatArray(shortBuffer.size)
    for (i in shortBuffer.indices) {
        floatArray[i] = shortBuffer[i] / 32768.0f
    }
    return floatArray
}
```

---

## 6. Performance & Memory Optimizations

### 1. Multi-threading
Configure intra-op threads based on available CPU cores:
```kotlin
val numCores = Runtime.getRuntime().availableProcessors()
val sessionOptions = OrtSession.SessionOptions().apply {
    setIntraOpNumThreads(Math.min(4, numCores))
    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
}
```

### 2. Memory-Mapping from Assets (Zero Copy)
Instead of reading the entire 117 MB into a byte array in RAM:
```kotlin
val assetFd = context.assets.openFd("asr/model.onnx")
val fileDescriptor = assetFd.fileDescriptor
val session = ortEnv.createSession(
    assetFd.createInputStream().channel.map(
        FileChannel.MapMode.READ_ONLY,
        assetFd.startOffset,
        assetFd.length
    )
)
```
This allows the Android OS kernel to page the model directly from the APK without consuming Java heap space.

---

## 7. C++ / NDK Integration (Advanced)

For ultra-low latency or smaller APK binary size, you can use the native C++ ONNX Runtime C API via JNI:

```cpp
#include <onnxruntime_c_api.h>

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
OrtEnv* env;
g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EthioASR", &env);

OrtSessionOptions* session_options;
g_ort->CreateSessionOptions(&session_options);
g_ort->SetIntraOpNumThreads(session_options, 4);

OrtSession* session;
g_ort->CreateSession(env, model_path, session_options, &session);
```

### Advantages of C++ NDK Pipeline:
1. **Zero Garbage Collection (GC) pauses**: Avoids creating short-lived `FloatBuffer` and `Array<FloatArray>` allocations.
2. **SIMD Vectorization**: Custom NEON assembly or C++ loops for zero-mean unit-variance normalization and CTC greedy search.
3. **Reduced APK Size**: Strip unused ONNX Runtime operators using the ONNX Runtime Mobile Operator Minimal Build Tool.
