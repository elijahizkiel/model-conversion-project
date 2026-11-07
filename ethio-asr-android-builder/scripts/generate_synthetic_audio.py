#!/usr/bin/env python3
"""generate_synthetic_audio.py
Create short synthetic wav files (3 seconds) for each target language.
The files are saved under ``datasets/`` with names ``<lang>.wav``.
"""
import os
import numpy as np
import soundfile as sf

# Target languages and a simple tone frequency for each (just for distinction)
LANGS = {
    "amharic": 440,   # A4
    "afaan_oro": 494, # B4
    "tigrinya": 554,  # C#5
    "sidama": 622,    # D#5
    "wolaytta": 698,  # F5
}

DURATION = 3.0  # seconds
SAMPLE_RATE = 16000

os.makedirs("datasets", exist_ok=True)

for lang, freq in LANGS.items():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # simple sine wave + a tiny amount of white noise
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.01 * np.random.randn(len(t))
    filename = os.path.join("datasets", f"{lang}.wav")
    sf.write(filename, audio.astype(np.float32), SAMPLE_RATE)
    print(f"Generated {filename} (freq {freq} Hz)")
