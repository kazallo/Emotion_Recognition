"""RAVDESS vorbereiten: ein Datensatz, der nur zum Testen benutzt wird.

RAVDESS wurde von anderen Schauspielern, in einem anderen Studio und mit
anderen Mikrofonen aufgenommen als CREMA-D. Kein Clip daraus wird je zum
Training benutzt. So lässt sich prüfen, ob ein Modell Gefühle in der Stimme
erkennt oder nur die Eigenheiten der CREMA-D-Aufnahmen gelernt hat.

Voraussetzung: data/Audio_Speech_Actors_01-24.zip
(von https://zenodo.org/records/1188976, ca. 200 MB)

Ergebnis: data/ravdess/*.wav und data/ravdess.csv, alle Clips mit split = "test"

Aufruf: uv run data_preprocessing/Ravdess_preprocessing.py
"""

import zipfile
from pathlib import PurePosixPath

import pandas as pd
import soundfile as sf

from Cremad_preprocessing import DATA_DIR, MIN_SECONDS, measure_audio

ZIP_PATH = DATA_DIR / "Audio_Speech_Actors_01-24.zip"
AUDIO_DIR = DATA_DIR / "ravdess"
OUTPUT = DATA_DIR / "ravdess.csv"

# Dritte Zahl im Dateinamen -> Name des Gefühls (dieselben Namen wie bei CREMA-D)
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
INTENSITIES = {"01": "normal", "02": "strong"}


def unpack() -> None:
    """Entpackt nur die WAV-Dateien, flach in einen Ordner."""
    if not ZIP_PATH.exists():
        raise SystemExit(f"{ZIP_PATH} fehlt. Bitte von https://zenodo.org/records/1188976 herunterladen.")
    AUDIO_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for member in archive.namelist():
            name = PurePosixPath(member).name
            target = AUDIO_DIR / name
            if name.endswith(".wav") and not target.exists():
                target.write_bytes(archive.read(member))


def read_files() -> pd.DataFrame:
    """Eine Zeile pro Datei mit den Angaben aus dem Dateinamen und der Dauer."""
    rows = []
    for file in sorted(AUDIO_DIR.glob("*.wav")):
        # Beispiel: 03-01-05-01-02-01-12
        # Modalität - Kanal - Gefühl - Intensität - Satz - Wiederholung - Sprecher
        _, _, emotion_code, intensity, sentence, repetition, actor = file.stem.split("-")
        info = sf.info(file)
        rows.append({
            "path": str(file.relative_to(DATA_DIR)),
            "speaker": int(actor),
            "sentence": sentence,
            "emotion": EMOTIONS[emotion_code],
            "intensity": INTENSITIES[intensity],
            "duration": info.duration,
            "sample_rate": info.samplerate,
        })
    return pd.DataFrame(rows)


def main() -> None:
    unpack()
    df = read_files()
    print(f"{len(df)} Dateien von {df['speaker'].nunique()} Sprechern gefunden")

    # Dieselbe Prüfung wie bei CREMA-D: Dauer nach dem Kürzen und Übersteuerung
    df = measure_audio(df)
    too_short = df["duration_trimmed"] < MIN_SECONDS
    if too_short.any():
        print(f"Aussortiert (nach dem Kürzen unter {MIN_SECONDS} s): {', '.join(df.loc[too_short, 'path'])}")
    df = df[~too_short]

    # Nur zum Testen: kein Clip aus RAVDESS kommt ins Training
    df = df.assign(split="test")
    df.to_csv(OUTPUT, index=False)

    print("\nClips pro Gefühl:")
    print(df["emotion"].value_counts())
    print(f"\nDauer vorher {df['duration'].mean():.2f} s, nach dem Kürzen {df['duration_trimmed'].mean():.2f} s (Mittelwert)")
    print(f"Abtastrate im Original: {sorted(df['sample_rate'].unique())} Hz (wird beim Laden auf 16 kHz umgerechnet)")
    print(f"\nGespeichert: {OUTPUT}")


if __name__ == "__main__":
    main()
