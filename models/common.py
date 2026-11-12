"""Gemeinsame Einstellungen und das Laden der Daten für alle Modell-Skripte."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

# Die Gefühle, mit denen trainiert wird. Hier eins entfernen oder hinzufügen,
# dann passen sich alle Skripte an. Ekel (disgust) ist bewusst nicht dabei.
EMOTIONS = ["anger", "joy", "neutral", "sadness"]

SEED = 42


def get_device() -> str:
    """Wählt automatisch NVIDIA-GPU (cuda auf Hochschul-Server), Apple-GPU (mps) oder CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

RESULTS_DIR = ROOT / "results"


def per_corpus_uar(table: pd.DataFrame, mask: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """UAR getrennt pro Korpus (nur für die Clips in `mask`).

    ESD (gespielt, sauber) und MELD (Fernsehen, verrauscht) sind sehr verschieden.
    Ein einziger Gesamtwert versteckt, wo das Modell wirklich schwach ist.
    """
    from sklearn.metrics import recall_score

    corpus = table.loc[mask, "corpus"].to_numpy()
    return {
        c: float(recall_score(y_true[corpus == c], y_pred[corpus == c], labels=np.unique(y_true[corpus == c]), average="macro"))
        for c in sorted(set(corpus))
    }


def save_results(name: str, results: dict) -> None:
    """Schreibt Kennzahlen eines Laufs nach results/<name>.json (zum Vergleichen und Nachschlagen)."""
    import json

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Kennzahlen: results/{name}.json")
