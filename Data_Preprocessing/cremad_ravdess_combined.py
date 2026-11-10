import argparse
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
 
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
 
CREMAD_RE = re.compile(r"^(\d{4})_[A-Za-z0-9]+_([A-Z]{3})_")
RAVDESS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")
 
CREMAD_EMOTIONS = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "joy", "NEU": "neutral", "SAD": "sadness"}
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "joy", "04": "sadness",
    "05": "anger", "06": "fear", "07": "disgust", "08": "surprised",
}
KEEP = ["anger", "joy", "neutral", "sadness"]
 
 
def scan() -> pd.DataFrame:
    rows = []
    for wav in sorted(DATA_DIR.rglob("*.wav")):
        rel = wav.relative_to(DATA_DIR).as_posix()
        if m := CREMAD_RE.match(wav.stem):
            rows.append((rel, CREMAD_EMOTIONS.get(m[2]), f"cremad_{m[1]}", "cremad"))
        elif m := RAVDESS_RE.match(wav.stem):
            modality, channel, emotion, *_, actor = m.groups()
            if modality == "03" and channel == "01":  # nur Audio, nur Sprache (kein Gesang)
                rows.append((rel, RAVDESS_EMOTIONS.get(emotion), f"ravdess_{actor}", "ravdess"))
    return pd.DataFrame(rows, columns=["path", "emotion", "speaker", "corpus"])
 
 
def assign_splits(table: pd.DataFrame, seed: int, val: float, test: float) -> pd.Series:
    """Sprecherunabhängig, pro Korpus: jeder Sprecher landet in genau einem Split."""
    rng = np.random.default_rng(seed)
    split_of = {}
    for corpus, group in table.groupby("corpus"):
        speakers = np.array(sorted(group["speaker"].unique()))
        rng.shuffle(speakers)
        n_test, n_val = round(len(speakers) * test), round(len(speakers) * val)
        for s in speakers[:n_test]:
            split_of[s] = "test"
        for s in speakers[n_test : n_test + n_val]:
            split_of[s] = "val"
        for s in speakers[n_test + n_val :]:
            split_of[s] = "train"
    return table["speaker"].map(split_of)
 
 
def main(name: str, seed: int, keep_all: bool) -> None:
    table = scan()
    if table.empty:
        raise SystemExit(f"Keine passenden .wav-Dateien in {DATA_DIR} gefunden.")
    print(f"Gefunden: {len(table)} Clips")
 
    if not keep_all:
        table = table[table["emotion"].isin(KEEP)].reset_index(drop=True)
        print(f"Nach Filter auf {KEEP}: {len(table)} Clips")
 
    table["split"] = assign_splits(table, seed, val=0.15, test=0.15)
    table = table[["path", "emotion", "split", "speaker", "corpus"]]
 
    print("\nClips pro Korpus und Emotion:")
    print(pd.crosstab(table["corpus"], table["emotion"]).to_string())
    print("\nClips pro Korpus und Split:")
    print(pd.crosstab(table["corpus"], table["split"]).to_string())
 
    table.to_csv(DATA_DIR / f"{name}.csv", index=False)
    print(f"\nGespeichert: data/{name}.csv")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-all", action="store_true", help="auch calm, disgust und surprised behalten")
    args = parser.parse_args()
    main(args.name, args.seed, args.keep_all)