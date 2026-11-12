import numpy as np
import soundfile as sf
import librosa

def load_and_preprocess_audio(audio_path, target_sr=16000, do_normalize=True):
    """
    Load an audio file, convert stereo to mono, resample to target_sr,
    and apply zero-mean unit-variance normalization.
    """
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

    wav = wav.astype(np.float32)

    if do_normalize:
        mean = np.mean(wav)
        std = np.std(wav)
        if std > 1e-7:
            wav = (wav - mean) / std

    return np.expand_dims(wav, axis=0)  # [1, sequence_length]
