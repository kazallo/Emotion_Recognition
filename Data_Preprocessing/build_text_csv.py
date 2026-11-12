"""Baut data/text_combined.csv aus 4 reinen Text-Emotions-Datensätzen (ohne MELD).

Umgesetzte Verbesserungen:
  1. Kein MELD (keine Audio/Video-Kontamination der Text-Labels).
  2. Gesprochene Dialog-Datensätze ergänzt (passen viel besser zu gesprochenen Sätzen auf der KI-Con):
       - dailydialog (echte englische Alltagsgespräche: anger, joy, neutral, sadness)
       - emocontext  (SemEval-2019 Task 3 Dialog-Äußerungen: angry, happy, sad, others->neutral)
  3. Herkunfts-Artefakte ("Shortcuts") bereinigt:
       - `neutral` kommt jetzt aus 3 verschiedenen Korpora (dailydialog, emocontext, goemotions),
         nicht mehr nur aus GoEmotions.
       - Aus CARER werden triviale Sätze entfernt, die das Label-Wort direkt verraten ("i feel sad"),
         und starre "i feel "-Präfixe teilweise bereinigt, damit das Modell keinen Korpus-Stil auswendig lernt.
  4. Gestrafftes GoEmotions-Mapping:
       - Schwammige Kategorien wie `approval` (Zustimmung != Freude) oder `disapproval` (Ablehnung != Wut)
         wurden entfernt; nur eindeutige Kern-Emotionen bleiben drin.
  5. Zusätzliche Spalte `valence` (-1.0 negativ, 0.0 neutral, +1.0 positiv) für das Valenz-Teilmodell.

Ergebnis: data/text_combined.csv mit den Spalten: text, emotion, valence, split, corpus

Aufruf: uv run data_preprocessing/build_text_csv.py
"""

import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KEEP = ["anger", "joy", "neutral", "sadness"]
VALENCE_MAP = {"anger": -1.0, "sadness": -1.0, "neutral": 0.0, "joy": 1.0}

# Gestrafftes GoEmotions-Mapping: nur echte Kern-Emotionen (ohne approval/disapproval/gratitude)
GOEMOTIONS_MAP = {
    2: "anger",      # anger
    3: "anger",      # annoyance
    17: "joy",       # joy
    1: "joy",        # amusement
    13: "joy",       # excitement
    18: "joy",       # love
    25: "sadness",   # sadness
    9: "sadness",    # disappointment
    16: "sadness",   # grief
    24: "sadness",   # remorse
    27: "neutral",   # neutral
}

# dair-ai/emotion (CARER): 0=sadness, 1=joy, 3=anger
CARER_MAP = {0: "sadness", 1: "joy", 3: "anger"}
CARER_LEAK_RE = re.compile(r"\b(sad|sadness|angry|anger|joy|joyful|happy|happiness)\b", re.IGNORECASE)
CARER_PREFIX_RE = re.compile(r"^(i\s+feel\s+|i\s+am\s+feeling\s+|im\s+feeling\s+)(that\s+|like\s+)?", re.IGNORECASE)

# DailyDialog: 0=neutral, 1=anger, 4=joy, 5=sadness
DAILYDIALOG_MAP = {0: "neutral", 1: "anger", 4: "joy", 5: "sadness"}

# EmoContext (SemEval-2019 Task 3)
EMOCONTEXT_MAP = {"angry": "anger", "happy": "joy", "sad": "sadness", "others": "neutral"}


def load_carer() -> pd.DataFrame:
    """Lädt dair-ai/emotion und entfernt triviale Shortcut-Sätze sowie starre 'i feel'-Anfänge."""
    split_files = {
        "train": "split/train-00000-of-00001.parquet",
        "val": "split/validation-00000-of-00001.parquet",
        "test": "split/test-00000-of-00001.parquet",
    }
    rows = []
    for split, filename in split_files.items():
        path = hf_hub_download(repo_id="dair-ai/emotion", filename=filename, repo_type="dataset")
        df = pd.read_parquet(path)
        df["emotion"] = df["label"].map(CARER_MAP)
        df = df[df["emotion"].isin(KEEP)].copy()
        # Entferne Sätze, die das Labelwort direkt enthalten
        df = df[~df["text"].str.contains(CARER_LEAK_RE, na=False)].copy()
        # Entferne bei der Hälfte der Sätze das verräterische "i feel "-Präfix
        cleaned = []
        for idx, txt in enumerate(df["text"].astype(str)):
            if idx % 2 == 0:
                txt = CARER_PREFIX_RE.sub("", txt).strip()
            cleaned.append(txt)
        df["text"] = cleaned
        df = df[df["text"].str.split().str.len() >= 4].copy()
        df["split"] = split
        df["corpus"] = "carer"
        rows.append(df[["text", "emotion", "split", "corpus"]])
    return pd.concat(rows, ignore_index=True)


def load_goemotions() -> pd.DataFrame:
    """Lädt Google GoEmotions mit gestrafftem Mapping."""
    split_files = {
        "train": "simplified/train-00000-of-00001.parquet",
        "val": "simplified/validation-00000-of-00001.parquet",
        "test": "simplified/test-00000-of-00001.parquet",
    }
    rows = []
    for split, filename in split_files.items():
        path = hf_hub_download(repo_id="google-research-datasets/go_emotions", filename=filename, repo_type="dataset")
        df = pd.read_parquet(path)
        mapped = []
        for labels in df["labels"]:
            emotions = {GOEMOTIONS_MAP[int(l)] for l in labels if int(l) in GOEMOTIONS_MAP}
            mapped.append(next(iter(emotions)) if len(emotions) == 1 else None)
        df["emotion"] = mapped
        df = df[df["emotion"].isin(KEEP)].copy()
        df["split"] = split
        df["corpus"] = "goemotions"
        rows.append(df[["text", "emotion", "split", "corpus"]])
    return pd.concat(rows, ignore_index=True)


def load_dailydialog() -> pd.DataFrame:
    """Lädt gesprochene Alltags-Äußerungen aus DailyDialog."""
    split_files = {
        "train": "data/train-00000-of-00001.parquet",
        "val": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet",
    }
    rows = []
    for split, filename in split_files.items():
        path = hf_hub_download(repo_id="pixelsandpointers/better_daily_dialog", filename=filename, repo_type="dataset")
        df = pd.read_parquet(path)
        df["emotion"] = df["emotion"].map(DAILYDIALOG_MAP)
        df = df[df["emotion"].isin(KEEP)].rename(columns={"utterance": "text"}).copy()
        # DailyDialog besteht zu >80 % aus neutral; wir begrenzen neutral auf die Anzahl von joy
        non_neutral = df[df["emotion"] != "neutral"]
        n_cap = max(len(non_neutral), 1500)
        neutral_sub = df[df["emotion"] == "neutral"].sample(
            n=min(len(df[df["emotion"] == "neutral"]), n_cap), random_state=42
        )
        df = pd.concat([non_neutral, neutral_sub], ignore_index=True)
        df["split"] = split
        df["corpus"] = "dailydialog"
        rows.append(df[["text", "emotion", "split", "corpus"]])
    return pd.concat(rows, ignore_index=True)


def load_emocontext() -> pd.DataFrame:
    """Lädt Dialog-Äußerungen aus SemEval-2019 Task 3 (EmoContext)."""
    split_files = {
        "train": "cleansed_train.csv",
        "val": "cleansed_dev.csv",
        "test": "cleansed_test.csv",
    }
    rows = []
    for split, filename in split_files.items():
        path = hf_hub_download(repo_id="oneonlee/cleansed_emocontext", filename=filename, repo_type="dataset")
        df = pd.read_csv(path)
        df["emotion"] = df["label"].astype(str).str.strip().str.lower().map(EMOCONTEXT_MAP)
        df = df[df["emotion"].isin(KEEP)].rename(columns={"turn3": "text"}).copy()
        # Auch in EmoContext ist 'others' (neutral) im Training stark überrepräsentiert -> balancieren
        if split == "train":
            non_neutral = df[df["emotion"] != "neutral"]
            neutral_sub = df[df["emotion"] == "neutral"].sample(
                n=min(len(df[df["emotion"] == "neutral"]), len(non_neutral) // 2), random_state=42
            )
            df = pd.concat([non_neutral, neutral_sub], ignore_index=True)
        df["split"] = split
        df["corpus"] = "emocontext"
        rows.append(df[["text", "emotion", "split", "corpus"]])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print("1/4 Lade CARER (bereinigt von Label-Wörtern & 'i feel'-Artefakten)...")
    carer = load_carer()
    print("2/4 Lade Google GoEmotions (gestrafftes Mapping)...")
    goemo = load_goemotions()
    print("3/4 Lade DailyDialog (gesprochene Alltags-Sätze)...")
    daily = load_dailydialog()
    print("4/4 Lade EmoContext (SemEval-2019 Dialog-Äußerungen)...")
    emoctx = load_emocontext()

    table = pd.concat([carer, goemo, daily, emoctx], ignore_index=True)
    table["text"] = table["text"].fillna("").astype(str).str.strip()
    table = table[table["text"].str.len() >= 3].reset_index(drop=True)
    table["valence"] = table["emotion"].map(VALENCE_MAP).astype("float32")

    print("\nSätze pro Korpus und Emotion:")
    print(pd.crosstab(table["corpus"], table["emotion"]).to_string())
    print("\nSätze pro Korpus und Split:")
    print(pd.crosstab(table["corpus"], table["split"]).to_string())

    out = DATA_DIR / "text_combined.csv"
    table.to_csv(out, index=False)
    print(f"\nGespeichert: data/text_combined.csv ({len(table)} Sätze)")


if __name__ == "__main__":
    main()
