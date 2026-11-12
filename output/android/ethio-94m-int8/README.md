# ethio-94m (INT8) Android Deployment Package

This directory contains deployment assets for offline speech recognition across 5 Ethiopian languages:
- Amharic (amh)
- Afaan Oromo (orm)
- Tigrinya (tir)
- Sidama (sid)
- Wolaytta (wal)

## Files
- `model.onnx`: INT8 ONNX model (117.00 MB)
- `vocab.json`: CTC vocabulary mapping token IDs to characters
- `manifest.json`: Specification metadata (sample rate, inputs, CTC blank ID)
- `SHA256SUMS`: Cryptographic verification hashes

## Quick Android Integration (Kotlin)
1. Place the entire folder or files into `app/src/main/assets/asr/`.
2. Add dependency in `build.gradle.kts`:
   ```kotlin
   implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
   ```
3. Load the session:
   ```kotlin
   val env = OrtEnvironment.getEnvironment()
   val modelBytes = context.assets.open("asr/model.onnx").readBytes()
   val session = env.createSession(modelBytes)
   ```
4. Feed normalized 16kHz mono audio float buffer into tensor `input_values`.
5. Greedy CTC decode the output `logits` using `vocab.json` (collapse repeated tokens, filter ID 408).
