"""Ein fertig trainiertes Stimm-Modell auf einem Datensatz testen, ohne neu zu trainieren.

Damit lässt sich prüfen, wie gut ein auf CREMA-D trainiertes Modell auf fremden
Daten (z. B. RAVDESS) funktioniert. Wichtig: Die Normalisierung kommt aus dem
Training und wird nicht an die neuen Daten angepasst. Sonst würde sich das
Modell heimlich an die Testdaten gewöhnen.

Modelle:
  svm_raw       SVM auf den 42 Merkmalen            (models/train_svm.py --raw)
  svm_encoded   Autoencoder + SVM                   (models/autoencoder.py, models/train_svm.py)
  svm_wav2vec   SVM auf wav2vec 2.0                 (models/train_svm.py --wav2vec)
  mlp           neuronales Netz auf wav2vec 2.0     (models/train_mlp.py)

Aufruf: uv run models/evaluate.py ravdess svm_raw
        uv run models/evaluate.py ravdess mlp
"""

import argparse

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score

from autoencoder import StackedAutoencoder
from common import CHECKPOINT_DIR, EMOTIONS, load_dataset
from train_mlp import EmotionMLP
from train_svm import wav2vec_layer


def predict_svm(svm_name: str, features: np.ndarray, encode: bool) -> np.ndarray:
    if encode:
        # Erst mit dem gespeicherten Autoencoder verdichten, mit der Normalisierung aus dem Training
        saved = torch.load(CHECKPOINT_DIR / "autoencoder.pt", weights_only=False)
        autoencoder = StackedAutoencoder(saved["n_inputs"])
        autoencoder.load_state_dict(saved["state"])
        autoencoder.eval()
        with torch.no_grad():
            x = torch.from_numpy((features - saved["mean"]) / saved["std"]).float()
            features = autoencoder.encoder(x).numpy()
    svm = joblib.load(CHECKPOINT_DIR / f"{svm_name}.joblib")
    return svm.predict(features)


def predict_mlp(embeddings: np.ndarray, embedding: str) -> np.ndarray:
    saved = torch.load(CHECKPOINT_DIR / f"mlp_{embedding}.pt", weights_only=False)
    model = EmotionMLP(embeddings.shape[2], len(saved["emotions"]))
    model.load_state_dict(saved["state"])
    model.eval()
    x = (embeddings[:, saved["layer"]].astype("float32") - saved["mean"]) / saved["std"]
    with torch.no_grad():
        classes = model(torch.from_numpy(x)).argmax(dim=1).numpy()
    return np.array(saved["emotions"])[classes]


def main(dataset: str, model: str) -> None:
    is_mlp = model.startswith("mlp_")
    embedding = model.removeprefix("mlp_")  # "mlp_hubert" -> "hubert"

    if is_mlp:
        features_file = f"{dataset}_{embedding}.npy"
    elif model == "svm_wav2vec":
        features_file = f"{dataset}_wav2vec.npy"
    else:
        features_file = f"{dataset}_features.npy"
    table, features, usable = load_dataset(dataset, features_file)
    test = usable & (table["split"] == "test").to_numpy()
    y_true = table.loc[test, "emotion"].to_numpy()

    if is_mlp:
        prediction = predict_mlp(features[test], embedding)
    elif model == "svm_wav2vec":
        layer_features = features[test][:, wav2vec_layer()].astype("float32")
        prediction = predict_svm(model, layer_features, encode=False)
    else:
        prediction = predict_svm(model, features[test], encode=(model == "svm_encoded"))

    labels = [e for e in EMOTIONS if e in set(y_true)]
    print(f"Modell: {model}, getestet auf '{dataset}' ({test.sum()} Clips)\n")
    print(f"Trefferquote pro Gefühl gemittelt (UAR): {recall_score(y_true, prediction, average='macro'):.1%}")
    print(f"Einfache Trefferquote: {np.mean(prediction == y_true):.1%}\n")
    for emotion, r in zip(labels, recall_score(y_true, prediction, labels=labels, average=None)):
        print(f"  {emotion:<8} {r:.1%}")

    print("\nVerwechslungsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    print(f"{'':<9}" + "".join(f"{e:>9}" for e in labels))
    for emotion, row in zip(labels, confusion_matrix(y_true, prediction, labels=labels)):
        print(f"{emotion:<9}" + "".join(f"{n:>9}" for n in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="z. B. ravdess oder cremad")
    parser.add_argument("model", choices=["mlp_hubert", "mlp_wav2vec", "mlp_preprocessed_hubert", "mlp_preprocessed_wav2vec", "svm_hubert", "svm_wav2vec", "svm_encoded", "svm_raw"])
    args = parser.parse_args()
    main(args.dataset, args.model)
