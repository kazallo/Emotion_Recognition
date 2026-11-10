"""Alles in einer Datei: sammelt CREMA-D, RAVDESS, ESD und MELD aus data/ ein und schreibt data/combined.csv.

Was passiert:
  1. CREMA-D, RAVDESS, ESD: .wav-Dateien werden am Dateinamen erkannt (Ordnernamen sind egal).
  2. MELD: Labels aus *_sent_emo.csv, Audio wird mit ffmpeg aus den .mp4 extrahiert
     (16 kHz, mono) nach data/meld_audio/. Videos ohne Ton und stumme Clips werden aussortiert.
  3. Nur die Emotionen in KEEP bleiben übrig, einheitlich benannt.
  4. Splits: CREMA-D, RAVDESS, ESD nach Sprechern (kein Sprecher in zwei Splits),
     MELD mit den offiziellen Splits (train / dev -> val / test).
  5. Ausgabe: data/combined.csv mit den Spalten path, emotion, split, speaker, corpus, text.

Voraussetzung für MELD: ffmpeg (Mac: brew install ffmpeg)

Aufruf:
    uv run data_preprocessing/build_combined_csv.py
    uv run data_preprocessing/build_combined_csv.py --train-only esd
"""
import argparse
import random
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MELD_AUDIO = DATA_DIR / "meld_audio"

KEEP = ["anger", "joy", "neutral", "sadness"]
COLUMNS = ["path", "emotion", "split", "speaker", "corpus", "text"]

CREMAD_RE = re.compile(r"^(\d{4})_[A-Za-z0-9]+_([A-Z]{3})_")
RAVDESS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")
ESD_RE = re.compile(r"^(\d{4})_(\d{6})$")
MELD_RE = re.compile(r"^dia(\d+)_utt(\d+)$")

CREMAD_EMOTIONS = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "joy", "NEU": "neutral", "SAD": "sadness"}
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "joy", "04": "sadness",
    "05": "anger", "06": "fear", "07": "disgust", "08": "surprised",
}
ESD_EMOTIONS = {"angry": "anger", "happy": "joy", "neutral": "neutral", "sad": "sadness", "surprise": "surprised"}
MELD_CSV_SPLITS = {"train": "train", "dev": "val", "test": "test"}  # Anfang des CSV-Namens -> Split
MELD_FOLDER = {"train": "train", "val": "dev", "test": "test"}


def make_row(rel: Path, emotion, speaker, corpus, split=None, text=""):
    return {"path": rel.as_posix(), "emotion": emotion, "split": split, "speaker": speaker, "corpus": corpus, "text": text}


# ------------------------------------------------------------------ CREMA-D, RAVDESS, ESD
def scan_audio() -> list[dict]:
    rows = []
    for wav in sorted(DATA_DIR.rglob("*.wav")):
        rel, stem = wav.relative_to(DATA_DIR), wav.stem
        if m := CREMAD_RE.match(stem):
            rows.append(make_row(rel, CREMAD_EMOTIONS.get(m[2]), f"cremad_{m[1]}", "cremad"))
        elif m := RAVDESS_RE.match(stem):
            modality, channel, emotion, *_, actor = m.groups()
            if modality == "03" and channel == "01":  # nur Audio, nur Sprache (kein Gesang)
                rows.append(make_row(rel, RAVDESS_EMOTIONS.get(emotion), f"ravdess_{actor}", "ravdess"))
        elif (m := ESD_RE.match(stem)) and 11 <= int(m[1]) <= 20:  # 0011-0020 = englische Sprecher
            emotion = next((ESD_EMOTIONS[p.lower()] for p in rel.parts if p.lower() in ESD_EMOTIONS), None)
            rows.append(make_row(rel, emotion, f"esd_{m[1]}", "esd"))
    return rows


# ------------------------------------------------------------------ MELD
def split_of_path(rel: Path):
    """Split aus dem Ordnernamen (train_splits, dev_splits_complete, output_repeated_splits_test, ...)."""
    for part in reversed(rel.parts[:-1]):
        p = part.lower()
        if "train" in p:
            return "train"
        if "dev" in p:
            return "val"
        if "test" in p:
            return "test"
    return None


def to_wav(src: Path, dst: Path) -> bool:
    """Extrahiert die Tonspur als 16-kHz-Mono-WAV. False, wenn das Video keinen Ton hat oder defekt ist."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
        capture_output=True,
    )
    ok = result.returncode == 0 and dst.exists()
    if not ok:
        dst.unlink(missing_ok=True)
    return ok


def is_silent(path: Path, threshold: int = 100, min_seconds: float = 0.3) -> bool:
    """True, wenn der Clip praktisch stumm oder fast leer ist (16-Bit-PCM-WAV)."""
    try:
        with wave.open(str(path), "rb") as f:
            rate = f.getframerate()
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    except (wave.Error, EOFError):
        return False  # unbekanntes Format: nicht entscheiden
    return len(data) < min_seconds * rate or np.abs(data).max() < threshold


def scan_meld() -> list[dict]:
    csvs = {}
    for f in DATA_DIR.rglob("*_sent_emo.csv"):
        split = MELD_CSV_SPLITS.get(f.name.split("_")[0])
        if split:
            csvs[split] = f
    if not csvs:
        print("MELD: keine *_sent_emo.csv in data/ gefunden, übersprungen")
        return []

    # Mediendateien indexieren. Dialog-IDs beginnen in jedem Split bei 0, deshalb (Split, Name).
    # .wav kommt zuletzt und überschreibt .mp4, damit schon extrahierte Dateien wiederverwendet werden.
    index = {}
    for suffix in (".mp4", ".wav"):
        for f in DATA_DIR.rglob(f"*{suffix}"):
            if MELD_RE.match(f.stem) and (split := split_of_path(f.relative_to(DATA_DIR))):
                index[(split, f.stem)] = f

    rows, missing, failed, silent = [], 0, 0, 0
    for split, csv in csvs.items():
        table = pd.read_csv(csv)
        for _, r in tqdm(table.iterrows(), total=len(table), desc=f"MELD {split}"):
            emotion = str(r["Emotion"]).strip().lower()
            if emotion not in KEEP:  # unnötige Clips gar nicht erst konvertieren
                continue
            stem = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
            src = index.get((split, stem))
            if src is None:
                missing += 1
                continue
            if src.suffix == ".mp4":
                if shutil.which("ffmpeg") is None:
                    raise SystemExit("ffmpeg fehlt (Mac: brew install ffmpeg). Es extrahiert das Audio aus den MELD-Videos.")
                dst = MELD_AUDIO / MELD_FOLDER[split] / f"{stem}.wav"
                if not dst.exists() and not to_wav(src, dst):
                    failed += 1  # kein Ton oder defektes Video
                    continue
                src = dst
            if is_silent(src):
                silent += 1
                continue
            rows.append(make_row(src.relative_to(DATA_DIR), emotion, f"meld_{r['Speaker']}", "meld", split, str(r["Utterance"])))
    print(f"MELD: {missing} Clips ohne Mediendatei, {failed} ohne Ton oder defekt, {silent} stumm oder zu kurz")
    return rows


# ------------------------------------------------------------------ Splits
def assign_splits(table: pd.DataFrame, seed: int, val: float, test: float, train_only: list[str]) -> None:
    """Sprecherunabhängig und pro Korpus. Bereits gesetzte Splits (MELD) bleiben unverändert.
    Jeder Korpus hat einen eigenen Zufallsgenerator, ein neuer Korpus verändert die anderen Splits nicht."""
    for corpus, group in table[table["split"].isna()].groupby("corpus"):
        if corpus in train_only:
            table.loc[group.index, "split"] = "train"
            continue
        speakers = sorted(group["speaker"].unique())
        random.Random(f"{seed}-{corpus}").shuffle(speakers)
        n_test, n_val = round(len(speakers) * test), round(len(speakers) * val)
        mapping = {s: "test" if i < n_test else "val" if i < n_test + n_val else "train" for i, s in enumerate(speakers)}
        table.loc[group.index, "split"] = group["speaker"].map(mapping)


def main(name: str, seed: int, train_only: list[str]) -> None:
    table = pd.DataFrame(scan_audio() + scan_meld(), columns=COLUMNS)
    if table.empty:
        raise SystemExit(f"Keine passenden Dateien in {DATA_DIR} gefunden.")
    print(f"\nGefunden (vor Filter): {table['corpus'].value_counts().to_dict()}")

    table = table[table["emotion"].isin(KEEP)].reset_index(drop=True)
    print(f"Nach Filter auf {KEEP}: {len(table)} Clips")

    assign_splits(table, seed, val=0.15, test=0.15, train_only=train_only)
    table = table[COLUMNS].fillna({"text": ""})

    print("\nClips pro Korpus und Emotion:")
    print(pd.crosstab(table["corpus"], table["emotion"]).to_string())
    print("\nClips pro Korpus und Split:")
    print(pd.crosstab(table["corpus"], table["split"]).to_string())

    table.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"\nGespeichert: data/{name}.csv ({len(table)} Zeilen)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-only", nargs="*", default=[], help="Korpora, die komplett ins Training gehen (z. B. esd)")
    args = parser.parse_args()
    main(args.name, args.seed, args.train_only)
