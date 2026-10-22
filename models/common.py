"""Gemeinsame Einstellungen und das Laden der Daten für alle Modell-Skripte."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

# Die Gefühle, mit denen trainiert wird. Hier eins entfernen oder hinzufügen,
# dann passen sich alle Skripte an. Ekel (disgust) ist bewusst nicht dabei.
EMOTIONS = ["anger", "fear", "joy", "neutral", "sadness"]

SEED = 42


def load_dataset(dataset: str, features_file: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Lädt Tabelle und Merkmale und sagt, welche Clips benutzt werden.

    Gibt (Tabelle, Merkmale, Maske) zurück. Tabelle und Merkmale behalten alle
    Zeilen in derselben Reihenfolge wie data/<datensatz>.csv. Die Maske ist
    True für Clips mit einem der EMOTIONS und ohne fehlende Werte.
    """
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    features = np.load(DATA_DIR / features_file)
    assert len(table) == len(features), "Tabelle und Merkmale passen nicht zusammen!"

    # reshape: funktioniert für (Clips, 42) genauso wie für (Clips, Schichten, 1024)
    has_nan = np.isnan(features.reshape(len(features), -1)).any(axis=1)
    usable = table["emotion"].isin(EMOTIONS).to_numpy() & ~has_nan
    return table, features, usable