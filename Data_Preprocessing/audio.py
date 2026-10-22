from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000

# Fenster für die Lautstärkemessung: 25 ms lang, alle 10 ms ein neues
FRAME_MS = 25
HOP_MS = 10

# Einstellungen für das Abschneiden der Stille (an CREMA-D ausprobiert)
MARGIN_DB = 6    # so viel lauter als das Rauschen muss ein Fenster sein, um als Sprache zu gelten
PAD_MS = 200     # so viel Rand bleibt vor und nach der Sprache stehen
# Bei sehr sauberen Aufnahmen (z. B. RAVDESS) ist die Stille fast digital still, und schon
# leises Atmen läge über der Schwelle. Deshalb gilt zusätzlich: Sprache ist höchstens so
# viel leiser als die lauteste Stelle. Bei verrauschten Aufnahmen (CREMA-D) ändert das nichts.
MAX_BELOW_PEAK_DB = 40


def trim_silence(wave: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Schneidet Stille am Anfang und Ende ab, auch wenn im Hintergrund Rauschen ist.

    Gibt den gekürzten Clip und eine Maske zurück (True = Fenster enthält Sprache).
    """
    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)

    rms = librosa.feature.rms(y=wave, frame_length=frame, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)

    # Die leisesten 10 % der Fenster sind fast sicher Stille: daraus der Rauschpegel
    noise_floor = np.percentile(db, 10)
    speech = (db > noise_floor + MARGIN_DB) & (db > -MAX_BELOW_PEAK_DB)
    if not speech.any():
        return wave, speech

    first, last = np.flatnonzero(speech)[[0, -1]]
    pad = int(sr * PAD_MS / 1000)
    start = max(first * hop - pad, 0)
    end = min(last * hop + frame + pad, len(wave))
    return wave[start:end], speech


def load_audio(path: Path) -> np.ndarray:
    """Lädt einen Clip als Mono mit 16 kHz, ohne Gleichanteil und ohne Stille am Rand."""
    wave, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wave = wave - wave.mean()
    trimmed, _ = trim_silence(wave)
    return trimmed