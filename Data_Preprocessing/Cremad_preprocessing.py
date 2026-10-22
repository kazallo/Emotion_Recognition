from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from audio import SAMPLE_RATE, load_audio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "cremad"
OUTPUT = DATA_DIR / "cremad.csv"

# Kürzel im Dateinamen -> Name des Gefühls (Namen wie in MELD)
EMOTIONS = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "sadness",
}

# Clips, die nach dem Kürzen kürzer sind, werden aussortiert
MIN_SECONDS = 0.5

# Anteil der Sprecher in Validierung und Test, der Rest ist Training
VAL_SHARE = 0.1
TEST_SHARE = 0.2
SEED = 42


def read_files() -> pd.DataFrame:
    """Eine Zeile pro Datei mit den Angaben aus dem Dateinamen und der Dauer."""
    rows = []
    for file in sorted(AUDIO_DIR.glob("*.wav")):
        # Beispiel: 1001_DFA_ANG_XX -> Sprecher, Satz, Gefühl, Intensität
        speaker, sentence, emotion_code, intensity = file.stem.split("_")
        info = sf.info(file)
        rows.append({
            "path": str(file.relative_to(DATA_DIR)),
            "speaker": int(speaker),
            "sentence": sentence,
            "emotion": EMOTIONS[emotion_code],
            "intensity": intensity,
            "duration": info.duration,
            "sample_rate": info.samplerate,
        })
    return pd.DataFrame(rows)


def measure_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Dauer nach dem Kürzen und Anteil übersteuerter Messwerte pro Clip."""
    trimmed, clipped = [], []
    for path in tqdm(df["path"], desc="Clips prüfen"):
        wave = load_audio(DATA_DIR / path)
        trimmed.append(len(wave) / SAMPLE_RATE)
        clipped.append(np.mean(np.abs(wave) > 0.99))
    return df.assign(duration_trimmed=trimmed, clipped_share=clipped)


def split_by_speaker(df: pd.DataFrame) -> pd.DataFrame:
    """Ordnet jeden Sprecher zufällig, aber wiederholbar, genau einem Teil zu."""
    speakers = np.sort(df["speaker"].unique())
    np.random.default_rng(SEED).shuffle(speakers)

    n_test = round(len(speakers) * TEST_SHARE)
    n_val = round(len(speakers) * VAL_SHARE)
    split_of = {}
    for i, speaker in enumerate(speakers):
        split_of[speaker] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return df.assign(split=df["speaker"].map(split_of))


def main() -> None:
    df = read_files()
    print(f"{len(df)} Dateien von {df['speaker'].nunique()} Sprechern gefunden")

    df = measure_audio(df)
    too_short = df["duration_trimmed"] < MIN_SECONDS
    if too_short.any():
        print(f"Aussortiert (nach dem Kürzen unter {MIN_SECONDS} s): {', '.join(df.loc[too_short, 'path'])}")
    df = df[~too_short]

    df = split_by_speaker(df)

    # Prüfung: Jeder Sprecher steht in genau einem Teil
    assert (df.groupby("speaker")["split"].nunique() == 1).all(), "Ein Sprecher steht in mehreren Teilen!"

    df.to_csv(OUTPUT, index=False)

    print("\nClips pro Gefühl und Teil:")
    print(pd.crosstab(df["emotion"], df["split"], margins=True))
    print("\nSprecher pro Teil:")
    print(df.groupby("split")["speaker"].nunique())
    print(f"\nDauer vorher {df['duration'].mean():.2f} s, nach dem Kürzen {df['duration_trimmed'].mean():.2f} s (Mittelwert)")
    print(f"Clips mit Übersteuerung: {(df['clipped_share'] > 0).sum()}")
    print(f"\nGespeichert: {OUTPUT}")


if __name__ == "__main__":
    main()