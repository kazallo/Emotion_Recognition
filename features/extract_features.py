"""Die 42 Merkmale aus Aouani & Ben Ayed (2020) für jeden Clip berechnen.

Pro Fenster (25 ms, alle 10 ms ein neues):
  - 13 MFCC, 13 Delta-MFCC, 13 Delta-Delta-MFCC
  - ZCR (Zero Crossing Rate)
  - TEO (Teager Energy Operator)
  - HNR (Harmonics-to-Noise Ratio, über Praat)
Danach wird jedes Merkmal über alle Fenster eines Clips gemittelt.

Ergebnis: data/<datensatz>_features.npy mit der Form (Clips, 42),
in derselben Reihenfolge wie die Zeilen in data/<datensatz>.csv,
und data/feature_names.txt mit den Namen der 42 Spalten.

Aufruf: uv run features/extract_features.py cremad
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.append(str(ROOT / "data_preprocessing"))

from audio import SAMPLE_RATE, load_audio  # noqa: E402

# Dieselben Fenster wie beim Abschneiden der Stille: 25 ms lang, alle 10 ms
N_FFT = 400
HOP = 160
N_MFCC = 13
N_MELS = 40

# Praat setzt bei Fenstern ohne Stimme (z. B. "s", "f", Pausen) diesen Platzhalter
HNR_UNDEFINED = -200

FEATURE_NAMES = (
    [f"mfcc_{i}" for i in range(N_MFCC)]
    + [f"delta_{i}" for i in range(N_MFCC)]
    + [f"delta2_{i}" for i in range(N_MFCC)]
    + ["zcr", "teo", "hnr"]
)


def mfcc_features(wave: np.ndarray) -> np.ndarray:
    """39 Werte: Mittelwerte von MFCC, Delta und Delta-Delta."""
    mfcc = librosa.feature.mfcc(y=wave, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # Jede Matrix hat die Form (13, Fenster); Mittelwert über die Fenster
    return np.concatenate([mfcc.mean(axis=1), delta.mean(axis=1), delta2.mean(axis=1)])


def zcr_feature(wave: np.ndarray) -> float:
    """Wie oft das Signal pro Fenster die Nulllinie kreuzt, gemittelt."""
    zcr = librosa.feature.zero_crossing_rate(wave, frame_length=N_FFT, hop_length=HOP)
    return float(zcr.mean())


def teo_feature(wave: np.ndarray) -> float:
    """Teager-Energie: psi[n] = x[n]^2 - x[n-1] * x[n+1], pro Fenster gemittelt, dann über alle Fenster."""
    psi = wave[1:-1] ** 2 - wave[:-2] * wave[2:]
    frames = librosa.util.frame(psi, frame_length=N_FFT, hop_length=HOP)
    return float(frames.mean(axis=0).mean())


def hnr_feature(wave: np.ndarray) -> float:
    """Verhältnis von Klang zu Rauschen in dB, nur über Fenster mit Stimme gemittelt."""
    sound = parselmouth.Sound(wave, sampling_frequency=SAMPLE_RATE)
    hnr = sound.to_harmonicity(time_step=HOP / SAMPLE_RATE).values[0]
    voiced = hnr[hnr > HNR_UNDEFINED]
    # Ohne ein einziges stimmhaftes Fenster gibt es keinen sinnvollen Wert
    return float(voiced.mean()) if len(voiced) else np.nan


def extract(wave: np.ndarray) -> np.ndarray:
    """Alle 42 Merkmale eines Clips, in der Reihenfolge von FEATURE_NAMES."""
    return np.concatenate([mfcc_features(wave), [zcr_feature(wave), teo_feature(wave), hnr_feature(wave)]])


def main(dataset: str) -> None:
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    features = np.stack(
        [extract(load_audio(DATA_DIR / path)) for path in tqdm(table["path"], desc="Merkmale berechnen")]
    ).astype("float32")

    missing = np.isnan(features).any(axis=1)
    if missing.any():
        print(f"Achtung: {missing.sum()} Clips ohne stimmhafte Stelle (HNR fehlt): {', '.join(table.loc[missing, 'path'])}")

    np.save(DATA_DIR / f"{dataset}_features.npy", features)
    (DATA_DIR / "feature_names.txt").write_text("\n".join(FEATURE_NAMES))
    print(f"Gespeichert: data/{dataset}_features.npy mit Form {features.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name der Tabelle in data/, z. B. cremad")
    main(parser.parse_args().dataset)