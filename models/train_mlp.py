"""Kleines neuronales Netz auf den Embeddings (wav2vec / HuBERT) trainieren.

Zwei Modi:
  layer (Standard)  Für jede Schicht wird ein Netz trainiert. Die Schicht mit der besten
                    Trefferquote auf dem Validierungsteil gewinnt.
  mix               Ein einziges Netz bekommt alle Schichten und lernt selbst, wie stark
                    jede Schicht zählt (gewichteter Mix, wie im SUPERB-Benchmark).

Erst nach der Auswahl wird einmal auf dem Testteil gemessen, gesamt und pro Korpus.

Ergebnis: checkpoints/mlp_<embedding>[_mix].pt und results/mlp_<embedding>[_mix].json

Aufruf: uv run models/train_mlp.py combined --embedding hubert
        uv run models/train_mlp.py combined --embedding hubert --mode mix
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score
from torch import nn

from common import CHECKPOINT_DIR, EMOTIONS, SEED, load_dataset, per_corpus_uar, save_results

HIDDEN = 256
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 400
# Training endet, wenn die Validierung so viele Epochen lang nicht besser wurde
PATIENCE = 30


class EmotionMLP(nn.Module):
    """1024 Zahlen -> 256 -> ein Wert pro Gefühl."""

    def __init__(self, n_inputs: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),  # schaltet im Training zufällig Neuronen ab, gegen Auswendiglernen
            nn.Linear(HIDDEN, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerMixMLP(nn.Module):
    """Alle Schichten -> gelernter gewichteter Mix (softmax) -> EmotionMLP."""

    def __init__(self, n_layers: int, n_inputs: int, n_classes: int) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.head = EmotionMLP(n_inputs, n_classes)

    def layer_weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (Batch, Schichten, 1024)
        mixed = (x * self.layer_weights()[None, :, None]).sum(dim=1)
        return self.head(mixed)


def uar(y_true, y_pred) -> float:
    """Trefferquote pro Gefühl, gemittelt (Unweighted Average Recall)."""
    return recall_score(y_true, y_pred, average="macro")


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(x).softmax(dim=1).numpy()


def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    return predict_proba(model, x).argmax(axis=1)


def train(model: nn.Module, x_train, y_train, x_val, y_val) -> tuple[dict, float]:
    """Trainiert das Netz, gibt die besten Gewichte und deren Validierungs-UAR zurück."""
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Seltene Gefühle zählen im Fehler stärker (wie class_weight="balanced" bei der SVM)
    counts = torch.bincount(y_train, minlength=len(EMOTIONS)).float().clamp(min=1)
    loss_fn = nn.CrossEntropyLoss(weight=counts.sum() / (len(EMOTIONS) * counts))

    best_uar, best_state, epochs_without_gain = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_uar = uar(y_val.numpy(), predict(model, x_val))
        if val_uar > best_uar:
            best_uar, epochs_without_gain = val_uar, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain == PATIENCE:
                break
    return best_state, best_uar


def main(dataset: str, embedding: str, mode: str) -> None:
    table, embeddings, usable = load_dataset(dataset, f"{dataset}_{embedding}.npy")
    # Gefühle als Zahlen 0, 1, 2, ... in der Reihenfolge von EMOTIONS
    labels = torch.tensor([EMOTIONS.index(e) if e in EMOTIONS else -1 for e in table["emotion"]])
    masks = {s: usable & (table["split"] == s).to_numpy() for s in ("train", "val", "test")}
    print(f"Clips: {masks['train'].sum()} Training, {masks['val'].sum()} Validierung, {masks['test'].sum()} Test\n")
    n_layers, n_inputs = embeddings.shape[1], embeddings.shape[2]
    train_idx = np.flatnonzero(masks["train"])

    def normalize(layers: slice | int):
        # Normalisieren mit Werten nur aus dem Trainingsteil
        x = embeddings[:, layers].astype("float32")
        mean, std = x[train_idx].mean(axis=0), x[train_idx].std(axis=0) + 1e-6
        return torch.from_numpy((x - mean) / std), mean, std

    if mode == "layer":
        best = {"val_uar": -1.0}
        for layer in range(n_layers):
            x, mean, std = normalize(layer)
            torch.manual_seed(SEED)
            model = EmotionMLP(n_inputs, len(EMOTIONS))
            state, val_uar = train(model, x[masks["train"]], labels[masks["train"]], x[masks["val"]], labels[masks["val"]])
            print(f"Schicht {layer:>2}: Validierung {val_uar:.1%}")
            if val_uar > best["val_uar"]:
                best = {"val_uar": val_uar, "layer": layer, "state": state, "mean": mean, "std": std, "x": x}
        model = EmotionMLP(n_inputs, len(EMOTIONS))
        x_all, extra = best["x"], {"layer": best["layer"]}
        print(f"\nBeste Schicht: {best['layer']} (Validierung {best['val_uar']:.1%})")
    else:
        x_all, mean, std = normalize(slice(None))
        torch.manual_seed(SEED)
        model = LayerMixMLP(n_layers, n_inputs, len(EMOTIONS))
        state, val_uar = train(model, x_all[masks["train"]], labels[masks["train"]], x_all[masks["val"]], labels[masks["val"]])
        best = {"val_uar": val_uar, "state": state, "mean": mean, "std": std}
        model.load_state_dict(state)
        weights = model.layer_weights().detach().numpy()
        print(f"Validierung {val_uar:.1%}")
        print("Schichtgewichte: " + " ".join(f"{w:.2f}" for w in weights))
        extra = {"layer": None, "layer_weights": weights.tolist()}

    # Erst jetzt wird der Testteil benutzt
    model.load_state_dict(best["state"])
    y_test = labels[masks["test"]].numpy()
    prediction = predict(model, x_all[masks["test"]])

    present = [i for i in range(len(EMOTIONS)) if i in set(y_test)]
    test_uar = uar(y_test, prediction)
    by_corpus = per_corpus_uar(table, masks["test"], y_test, prediction)
    recalls = recall_score(y_test, prediction, labels=present, average=None)
    print(f"Test: Trefferquote pro Gefühl gemittelt (UAR) {test_uar:.1%}")
    print("  pro Korpus: " + ", ".join(f"{c} {v:.1%}" for c, v in by_corpus.items()) + "\n")
    for i, r in zip(present, recalls):
        print(f"  {EMOTIONS[i]:<8} {r:.1%}")

    print("\nVerwechslungsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    print(f"{'':<9}" + "".join(f"{EMOTIONS[i]:>9}" for i in present))
    for i, row in zip(present, confusion_matrix(y_test, prediction, labels=present)):
        print(f"{EMOTIONS[i]:<9}" + "".join(f"{n:>9}" for n in row))

    name = f"mlp_{embedding}" + ("_mix" if mode == "mix" else "")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {"state": best["state"], "mean": best["mean"], "std": best["std"], "emotions": EMOTIONS, "mode": mode, **extra},
        CHECKPOINT_DIR / f"{name}.pt",
    )
    print(f"\nGespeichert: checkpoints/{name}.pt")
    save_results(name, {
        "dataset": dataset, "mode": mode, "val_uar": best["val_uar"], "test_uar": test_uar,
        "test_uar_per_corpus": by_corpus,
        "recall": dict(zip([EMOTIONS[i] for i in present], map(float, recalls))), **extra,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="combined")
    parser.add_argument("--embedding", default="wav2vec", choices=["wav2vec", "hubert", "distilhubert"])
    parser.add_argument("--mode", default="layer", choices=["layer", "mix"])
    args = parser.parse_args()
    main(args.dataset, args.embedding, args.mode)
