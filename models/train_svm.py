"""SVM mit RBF-Kern trainieren und auswerten.

Standard: die 15 Zahlen aus dem Autoencoder (wie das beste System im Paper).
Mit --raw stattdessen die 42 Merkmale direkt, zum Vergleich.
Mit --wav2vec die wav2vec-Embeddings, und zwar die Schicht, die das neuronale
Netz gewählt hat (aus checkpoints/mlp_wav2vec.pt). So unterscheiden sich SVM
und Netz nur noch im Klassifikator.

C und gamma werden am Validierungsteil gewählt. Der Testteil wird erst ganz am
Ende einmal benutzt.

Ergebnis: checkpoints/svm_encoded.joblib (bzw. svm_raw.joblib, svm_wav2vec.joblib)

Aufruf: uv run models/train_svm.py cremad
        uv run models/train_svm.py cremad --raw
        uv run models/train_svm.py cremad --wav2vec
"""

import argparse

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import CHECKPOINT_DIR, EMOTIONS, SEED, load_dataset

C_VALUES = [0.1, 1, 10, 100]
GAMMA_VALUES = ["scale", 0.1, 0.01, 0.001]


def make_svm(c, gamma, probability=False):
    # class_weight="balanced": seltene Gefühle zählen im Training stärker
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=c, gamma=gamma, class_weight="balanced", probability=probability, random_state=SEED),
    )


def uar(y_true, y_pred) -> float:
    """Trefferquote pro Gefühl, gemittelt (Unweighted Average Recall)."""
    return recall_score(y_true, y_pred, average="macro")


FEATURE_FILES = {
    "raw": "{dataset}_features.npy",
    "encoded": "{dataset}_encoded.npy",
    "wav2vec": "{dataset}_wav2vec.npy",
}


def wav2vec_layer() -> int:
    """Die Schicht, die das neuronale Netz am Validierungsteil als beste gewählt hat."""
    return torch.load(CHECKPOINT_DIR / "mlp_wav2vec.pt", weights_only=False)["layer"]


def main(dataset: str, name: str) -> None:
    table, features, usable = load_dataset(dataset, FEATURE_FILES[name].format(dataset=dataset))
    if name == "wav2vec":
        layer = wav2vec_layer()
        features = features[:, layer].astype("float32")
        print(f"wav2vec-Schicht {layer}")

    def part(split):
        mask = usable & (table["split"] == split).to_numpy()
        return features[mask], table.loc[mask, "emotion"].to_numpy()

    x_train, y_train = part("train")
    x_val, y_val = part("val")
    x_test, y_test = part("test")
    print(f"Merkmale: {name} ({features.shape[1]} pro Clip)")
    print(f"Clips: {len(y_train)} Training, {len(y_val)} Validierung, {len(y_test)} Test\n")

    # C und gamma am Validierungsteil wählen
    best = (-1.0, None, None)
    for c in C_VALUES:
        for gamma in GAMMA_VALUES:
            svm = make_svm(c, gamma).fit(x_train, y_train)
            score = uar(y_val, svm.predict(x_val))
            print(f"C={c:<5} gamma={gamma!s:<6} Validierung {score:.1%}")
            if score > best[0]:
                best = (score, c, gamma)
    val_score, c, gamma = best
    print(f"\nBeste Einstellung: C={c}, gamma={gamma} (Validierung {val_score:.1%})")

    # Endgültiges Modell mit Wahrscheinlichkeiten (für die spätere Kombination mit dem Text)
    svm = make_svm(c, gamma, probability=True).fit(x_train, y_train)
    prediction = svm.predict(x_test)

    labels = [e for e in EMOTIONS if e in set(y_test)]
    print(f"\nTest: Trefferquote pro Gefühl gemittelt (UAR) {uar(y_test, prediction):.1%}")
    print(f"      einfache Trefferquote {np.mean(prediction == y_test):.1%}\n")
    recalls = recall_score(y_test, prediction, labels=labels, average=None)
    for emotion, r in zip(labels, recalls):
        print(f"  {emotion:<8} {r:.1%}")

    print("\nVerwechslungsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    matrix = confusion_matrix(y_test, prediction, labels=labels)
    print(f"{'':<9}" + "".join(f"{e:>9}" for e in labels))
    for emotion, row in zip(labels, matrix):
        print(f"{emotion:<9}" + "".join(f"{n:>9}" for n in row))

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    joblib.dump(svm, CHECKPOINT_DIR / f"svm_{name}.joblib")
    print(f"\nGespeichert: checkpoints/svm_{name}.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cremad")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--raw", action="store_true", help="die 42 Merkmale direkt statt der Autoencoder-Merkmale")
    choice.add_argument("--wav2vec", action="store_true", help="die wav2vec-Embeddings (Schicht des neuronalen Netzes)")
    args = parser.parse_args()
    main(args.dataset, "raw" if args.raw else "wav2vec" if args.wav2vec else "encoded")