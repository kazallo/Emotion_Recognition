"""Ein fertig trainiertes Stimm-Modell auf einem Datensatz (oder einzelnen Korpus wie MELD) testen.

Damit lässt sich prüfen, wie gut z. B. ein OHNE MELD trainiertes Modell auf MELD
(oder ein CREMA-D-Modell auf RAVDESS) funktioniert. Die Normalisierung stammt immer
aus dem Training und wird nicht an die Testdaten angepasst.

Aufruf:
  uv run models/evaluate.py combined best_hubert_ms_wav2vec_ms_prosody_nomeld --corpus meld
  uv run models/evaluate.py combined best_hubert_ms_wav2vec_ms_prosody_nomeld_spknorm --corpus meld
  uv run models/evaluate.py ravdess mlp_hubert
"""

import argparse

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score

from autoencoder import StackedAutoencoder
from common import CHECKPOINT_DIR, DATA_DIR, EMOTIONS, get_device, load_dataset, per_corpus_uar
from train_best import FusionNet, apply_speaker_normalization, predict_logits, softmax_np
from train_mlp import EmotionMLP, LayerMixMLP
from train_svm import wav2vec_layer


def predict_svm(svm_name: str, features: np.ndarray, encode: bool) -> np.ndarray:
    if encode:
        saved = torch.load(CHECKPOINT_DIR / "autoencoder.pt", weights_only=False)
        autoencoder = StackedAutoencoder(saved["n_inputs"])
        autoencoder.load_state_dict(saved["state"])
        autoencoder.eval()
        with torch.no_grad():
            x = torch.from_numpy((features - saved["mean"]) / saved["std"]).float()
            features = autoencoder.encoder(x).numpy()
    svm = joblib.load(CHECKPOINT_DIR / f"{svm_name}.joblib")
    return svm.predict(features)


def predict_mlp(embeddings: np.ndarray, checkpoint: str) -> np.ndarray:
    saved = torch.load(CHECKPOINT_DIR / f"{checkpoint}.pt", weights_only=False)
    if saved.get("layer") is None:
        model = LayerMixMLP(embeddings.shape[1], embeddings.shape[2], len(saved["emotions"]))
        x = (embeddings.astype("float32") - saved["mean"]) / saved["std"]
    else:
        model = EmotionMLP(embeddings.shape[2], len(saved["emotions"]))
        x = (embeddings[:, saved["layer"]].astype("float32") - saved["mean"]) / saved["std"]
    model.load_state_dict(saved["state"])
    model.eval()
    with torch.no_grad():
        classes = model(torch.from_numpy(x)).argmax(dim=1).numpy()
    return np.array(saved["emotions"])[classes]


def predict_best(dataset: str, checkpoint: str, table, eval_mask: np.ndarray) -> np.ndarray:
    saved = torch.load(CHECKPOINT_DIR / f"{checkpoint}.pt", weights_only=False)
    features = saved["features"]
    speaker_norm = saved.get("speaker_norm", False)
    temperature = saved.get("temperature", 1.0)
    eval_idx = np.flatnonzero(eval_mask)

    inputs = []
    for f, m, s in zip(features, saved["means"], saved["stds"]):
        raw = np.load(DATA_DIR / f"{dataset}_{f}.npy", mmap_mode="r")
        sub = np.array(raw[eval_idx], dtype="float16")
        del raw
        if sub.ndim == 2:
            sub = sub[:, None, :]
        if speaker_norm:
            apply_speaker_normalization(sub, table["speaker"].to_numpy()[eval_idx])
        m_np, s_np = m.numpy().astype("float32"), s.numpy().astype("float32")
        for start in range(0, len(sub), 2000):
            sub[start : start + 2000] = ((sub[start : start + 2000].astype("float32") - m_np) / s_np).astype("float16")
        inputs.append(torch.from_numpy(sub))

    device = get_device()
    n_layers = [d.shape[1] for d in inputs]
    widths = [d.shape[2] for d in inputs]
    logits_all = []
    for state in saved["states"]:
        model = FusionNet(n_layers, widths, len(saved["emotions"])).to(device)
        model.load_state_dict(state)
        logits_all.append(predict_logits(model, inputs, device))
    proba = softmax_np(np.mean(logits_all, axis=0), temperature)
    return np.array(saved["emotions"])[proba.argmax(axis=1)]


def main(dataset: str, model: str, corpus: str | None) -> None:
    if model.startswith("best_"):
        saved = torch.load(CHECKPOINT_DIR / f"{model}.pt", weights_only=False)
        table, _, usable = load_dataset(dataset, f"{dataset}_{saved['features'][0]}.npy")
        test = usable & (table["split"] == "test").to_numpy()
        if corpus:
            test = test & (table["corpus"] == corpus).to_numpy()
        y_true = table.loc[test, "emotion"].to_numpy()
        prediction = predict_best(dataset, model, table, test)
    else:
        is_mlp = model.startswith("mlp_")
        embedding = model.removeprefix("mlp_").removesuffix("_mix")
        if is_mlp:
            features_file = f"{dataset}_{embedding}.npy"
        elif model == "svm_wav2vec":
            features_file = f"{dataset}_wav2vec.npy"
        else:
            features_file = f"{dataset}_features.npy"
        table, features, usable = load_dataset(dataset, features_file)
        test = usable & (table["split"] == "test").to_numpy()
        if corpus:
            test = test & (table["corpus"] == corpus).to_numpy()
        y_true = table.loc[test, "emotion"].to_numpy()
        if is_mlp:
            prediction = predict_mlp(features[test], model)
        elif model == "svm_wav2vec":
            layer_features = features[test][:, wav2vec_layer()].astype("float32")
            prediction = predict_svm(model, layer_features, encode=False)
        else:
            prediction = predict_svm(model, features[test], encode=(model == "svm_encoded"))

    labels = [e for e in EMOTIONS if e in set(y_true)]
    target_desc = f"{dataset} (nur {corpus})" if corpus else dataset
    print(f"Modell: {model}, getestet auf '{target_desc}' ({test.sum()} Clips)\n")
    print(f"Trefferquote pro Gefühl gemittelt (UAR): {recall_score(y_true, prediction, average='macro'):.1%}")
    print(f"Einfache Trefferquote: {np.mean(prediction == y_true):.1%}\n")
    y_true_idx = np.array([EMOTIONS.index(e) for e in y_true])
    pred_idx = np.array([EMOTIONS.index(e) for e in prediction])
    print("Pro Korpus: " + ", ".join(f"{c} {v:.1%}" for c, v in per_corpus_uar(table, test, y_true_idx, pred_idx).items()) + "\n")
    for emotion, r in zip(labels, recall_score(y_true, prediction, labels=labels, average=None)):
        print(f"  {emotion:<8} {r:.1%}")

    print("\nVerwechslungsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    print(f"{'':<9}" + "".join(f"{e:>9}" for e in labels))
    for emotion, row in zip(labels, confusion_matrix(y_true, prediction, labels=labels)):
        print(f"{emotion:<9}" + "".join(f"{n:>9}" for n in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="z. B. combined, ravdess oder cremad")
    parser.add_argument("model", help="Name des Checkpoints in checkpoints/, z. B. best_hubert_ms_wav2vec_ms_prosody_nomeld")
    parser.add_argument("--corpus", default=None, help="Optional nur einen Teil-Korpus testen, z. B. meld")
    args = parser.parse_args()
    main(args.dataset, args.model, args.corpus)
