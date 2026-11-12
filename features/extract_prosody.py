"""Prosodie-Merkmale (Tonhöhe, Betonung, Sprechtempo, Stimmqualität) für jeden Clip berechnen.

Im Gegensatz zu MFCCs (die vor allem Vokale und Klangfarbe beschreiben) misst dieses Skript
gezielt die Sprechmelodie. Für Emotionen und besonders für Ironie/Sarkasmus sind genau diese
Größen entscheidend (z. B. monotone Stimme vs. übertriebene Tonhöhensprünge, gedehntes
Sprechtempo, Stimmzittern).

Berechnete Merkmale (20 Werte pro Clip):
  - Tonhöhe (Pitch / F0 über Praat):
      f0_mean, f0_std, f0_min, f0_max, f0_range, f0_iqr (Spannweite & Schwankung),
      f0_slope (lineare Steigung über die Zeit: geht die Stimme hoch oder runter?),
      f0_end_diff (letztes Viertel minus Gesamtmittel: Satzende-Betonung)
  - Lautstärke (Energy / RMS):
      rms_mean, rms_std, rms_range, rms_max_ratio (Spitzenbetonung),
      rms_slope (wird der Sprecher lauter oder leiser?)
  - Sprechtempo & Rhythmus:
      duration (Clip-Dauer in Sekunden nach Stille-Schnitt),
      voiced_fraction (Anteil stimmhafter Fenster),
      speaking_rate (Anzahl der Silben-/Lautstärke-Gipfel pro Sekunde)
  - Stimmqualität (über Praat):
      hnr_mean, hnr_std (Klang-zu-Rausch-Verhältnis: klar vs. behaucht/gepresst),
      jitter (Tonhöhenzittern), shimmer (Lautstärkezittern)

Ergebnis: data/<datensatz>_prosody.npy mit Form (Clips, 20) und data/prosody_names.txt

Aufruf: uv run features/extract_prosody.py combined
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.append(str(ROOT / "data_preprocessing"))

from audio import SAMPLE_RATE, load_audio  # noqa: E402

HOP_S = 0.01  # 10 ms Schrittweite

PROSODY_NAMES = [
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_iqr", "f0_slope", "f0_end_diff",
    "rms_mean", "rms_std", "rms_range", "rms_max_ratio", "rms_slope",
    "duration", "voiced_fraction", "speaking_rate",
    "hnr_mean", "hnr_std", "jitter", "shimmer",
]


def linear_slope(y: np.ndarray) -> float:
    """Lineare Steigung einer Zeitreihe (normiert auf Länge 1)."""
    if len(y) < 2:
        return 0.0
    t = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(t, y, 1)[0])


def extract_prosody(wave: np.ndarray) -> np.ndarray:
    """Berechnet die 20 Prosodie-Merkmale für einen Audio-Clip."""
    duration = len(wave) / SAMPLE_RATE
    if duration < 0.15:
        return np.zeros(len(PROSODY_NAMES), dtype="float32")

    sound = parselmouth.Sound(wave, sampling_frequency=SAMPLE_RATE)

    # 1. Tonhöhe (Pitch / F0) in Halbtönen relativ zu 100 Hz (macht Werte sprecherunabhängiger)
    pitch = sound.to_pitch(time_step=HOP_S, pitch_floor=75.0, pitch_ceiling=500.0)
    f0_hz = pitch.selected_array["frequency"]
    voiced = f0_hz > 0
    voiced_fraction = float(voiced.mean()) if len(f0_hz) else 0.0

    if voiced.sum() >= 3:
        f0 = 12.0 * np.log2(f0_hz[voiced] / 100.0)  # Halbtöne
        f0_mean = float(f0.mean())
        f0_std = float(f0.std())
        f0_min, f0_max = float(np.percentile(f0, 5)), float(np.percentile(f0, 95))
        f0_range = f0_max - f0_min
        f0_iqr = float(np.percentile(f0, 75) - np.percentile(f0, 25))
        f0_slope = linear_slope(f0)
        quarter = max(1, len(f0) // 4)
        f0_end_diff = float(f0[-quarter:].mean() - f0_mean)
    else:
        f0_mean = f0_std = f0_min = f0_max = f0_range = f0_iqr = f0_slope = f0_end_diff = 0.0

    # 2. Lautstärke-Verlauf (RMS in dB)
    hop = int(SAMPLE_RATE * HOP_S)
    rms = librosa.feature.rms(y=wave, frame_length=400, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    rms_mean = float(rms_db.mean())
    rms_std = float(rms_db.std())
    rms_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    rms_max_ratio = float(rms.max() / (rms.mean() + 1e-8))
    rms_slope = linear_slope(rms_db)

    # 3. Sprechtempo (Silhengipfel pro Sekunde über geglätteter Lautstärke)
    peaks, _ = find_peaks(rms, distance=int(0.08 / HOP_S), prominence=rms.std() * 0.3)
    speaking_rate = float(len(peaks) / max(duration, 0.1))

    # 4. Stimmqualität (HNR, Jitter, Shimmer)
    hnr = sound.to_harmonicity(time_step=HOP_S).values[0]
    hnr_voiced = hnr[hnr > -150]
    hnr_mean = float(hnr_voiced.mean()) if len(hnr_voiced) else 0.0
    hnr_std = float(hnr_voiced.std()) if len(hnr_voiced) > 1 else 0.0

    try:
        point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 500.0)
        jitter = float(call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3))
        shimmer = float(call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6))
        if np.isnan(jitter):
            jitter = 0.0
        if np.isnan(shimmer):
            shimmer = 0.0
    except Exception:
        jitter = shimmer = 0.0

    return np.array([
        f0_mean, f0_std, f0_min, f0_max, f0_range, f0_iqr, f0_slope, f0_end_diff,
        rms_mean, rms_std, rms_range, rms_max_ratio, rms_slope,
        duration, voiced_fraction, speaking_rate,
        hnr_mean, hnr_std, jitter, shimmer,
    ], dtype="float32")


def main(dataset: str) -> None:
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    features = np.stack(
        [extract_prosody(load_audio(DATA_DIR / path)) for path in tqdm(table["path"], desc="Prosodie")]
    ).astype("float32")
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(DATA_DIR / f"{dataset}_prosody.npy", features)
    (DATA_DIR / "prosody_names.txt").write_text("\n".join(PROSODY_NAMES))
    print(f"Gespeichert: data/{dataset}_prosody.npy mit Form {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="combined")
    main(parser.parse_args().dataset)
