package com.ethioasr.recognizer

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.InputStream
import java.nio.FloatBuffer

/**
 * High-performance, offline ASR Recognizer for Ethiopian languages
 * (Amharic, Afaan Oromo, Tigrinya, Sidama, Wolaytta).
 */
class EthioASRRecognizer(private val context: Context, private val modelAssetPath: String = "asr/model.onnx") {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocab: Map<Int, String>
    private val blankTokenId: Int = 408
    private val sampleRate: Int = 16000

    init {
        // Load ONNX Model from assets
        val modelBytes = context.assets.open(modelAssetPath).use { it.readBytes() }
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4) // Utilize multi-core mobile CPUs
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = ortEnv.createSession(modelBytes, sessionOptions)

        // Load Vocabulary mapping
        val vocabJson = context.assets.open("asr/vocab.json").use { it.bufferedReader().readText() }
        val jsonObject = JSONObject(vocabJson)
        val map = mutableMapOf<Int, String>()
        jsonObject.keys().forEach { charKey ->
            val id = jsonObject.getInt(charKey)
            map[id] = charKey
        }
        vocab = map
    }

    /**
     * Preprocesses audio: Mono 16kHz PCM audio, zero-mean unit-variance normalized.
     */
    fun preprocessAudio(rawSamples: FloatArray): FloatArray {
        var sum = 0.0
        for (v in rawSamples) sum += v
        val mean = (sum / rawSamples.size).toFloat()

        var sumSq = 0.0
        for (v in rawSamples) sumSq += (v - mean) * (v - mean)
        val std = Math.sqrt(sumSq / rawSamples.size).toFloat()

        val normalized = FloatArray(rawSamples.size)
        val epsilon = 1e-7f
        for (i in rawSamples.indices) {
            normalized[i] = if (std > epsilon) (rawSamples[i] - mean) / std else rawSamples[i]
        }
        return normalized
    }

    /**
     * Run Speech Recognition inference and return decoded Ethiopic / Latin script.
     */
    fun transcribe(audioSamples: FloatArray): String {
        val normalizedSamples = preprocessAudio(audioSamples)
        val shape = longArrayOf(1, normalizedSamples.size.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(normalizedSamples), shape)

        tensor.use { inTensor ->
            val inputs = mapOf("input_values" to inTensor)
            session.run(inputs).use { results ->
                @Suppress("UNCHECKED_CAST")
                val logits = results[0].value as Array<Array<FloatArray>> // [batch, frames, vocab]
                return greedyCTCDecode(logits[0])
            }
        }
    }

    /**
     * Fast CTC Greedy Decoder with token collapsing and blank-token removal.
     */
    private fun greedyCTCDecode(frames: Array<FloatArray>): String {
        val collapsedTokens = mutableListOf<Int>()
        var prevToken = -1

        for (frame in frames) {
            var maxIdx = 0
            var maxVal = frame[0]
            for (i in 1 until frame.size) {
                if (frame[i] > maxVal) {
                    maxVal = frame[i]
                    maxIdx = i
                }
            }

            if (maxIdx != prevToken) {
                collapsedTokens.add(maxIdx)
                prevToken = maxIdx
            }
        }

        val stringBuilder = StringBuilder()
        for (token in collapsedTokens) {
            if (token != blankTokenId) {
                val charStr = vocab[token] ?: ""
                // Replace CTC separator | with space if present
                if (charStr == "|") {
                    stringBuilder.append(" ")
                } else {
                    stringBuilder.append(charStr)
                }
            }
        }
        return stringBuilder.toString().trim()
    }

    fun close() {
        session.close()
    }
}
