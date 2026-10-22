"""Kleines neuronales Netz auf den wav2vec-Embeddings trainieren.

Für jede wav2vec-Schicht wird ein Netz trainiert. Die Schicht mit der besten
Trefferquote auf dem Validierungsteil gewinnt. Erst dann wird einmal auf dem
Testteil gemessen.

Ergebnis: checkpoints/mlp_wav2vec.pt (Netz, gewählte Schicht, Normalisierung)

Aufruf: uv run models/train_mlp.py cremad
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score
from torch import nn

from common import CHECKPOINT_DIR, EMOTIONS, SEED, load_dataset

HIDDEN = 256
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 100
# Training endet, wenn die Validierung so viele Epochen lang nicht besser wurde
PATIENCE = 10


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


def uar(y_true, y_pred) -> float:
    """Trefferquote pro Gefühl, gemittelt (Unweighted Average Recall)."""
    return recall_score(y_true, y_pred, average="macro")


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(x).argmax(dim=1).numpy()


def train(x_train, y_train, x_val, y_val) -> tuple[dict, float]:
    """Trainiert ein Netz, gibt die besten Gewichte und deren Validierungs-UAR zurück."""
    torch.manual_seed(SEED)
    model = EmotionMLP(x_train.shape[1], len(EMOTIONS))
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


def main(dataset: str) -> None:
    table, embeddings, usable = load_dataset(dataset, f"{dataset}_wav2vec.npy")
    # Gefühle als Zahlen 0, 1, 2, ... in der Reihenfolge von EMOTIONS
    labels = torch.tensor([EMOTIONS.index(e) if e in EMOTIONS else -1 for e in table["emotion"]])
    masks = {s: usable & (table["split"] == s).to_numpy() for s in ("train", "val", "test")}
    print(f"Clips: {masks['train'].sum()} Training, {masks['val'].sum()} Validierung, {masks['test'].sum()} Test\n")

    best = {"val_uar": -1.0}
    for layer in range(embeddings.shape[1]):
        x = embeddings[:, layer].astype("float32")
        # Normalisieren mit Werten nur aus dem Trainingsteil
        mean, std = x[masks["train"]].mean(axis=0), x[masks["train"]].std(axis=0) + 1e-6
        x = torch.from_numpy((x - mean) / std)
        state, val_uar = train(x[masks["train"]], labels[masks["train"]], x[masks["val"]], labels[masks["val"]])
        print(f"Schicht {layer:>2}: Validierung {val_uar:.1%}")
        if val_uar > best["val_uar"]:
            best = {"val_uar": val_uar, "layer": layer, "state": state, "mean": mean, "std": std, "x": x}

    # Erst jetzt, mit fester Schicht, wird der Testteil benutzt
    model = EmotionMLP(embeddings.shape[2], len(EMOTIONS))
    model.load_state_dict(best["state"])
    y_test = labels[masks["test"]].numpy()
    prediction = predict(model, best["x"][masks["test"]])

    present = [i for i in range(len(EMOTIONS)) if i in set(y_test)]
    print(f"\nBeste Schicht: {best['layer']} (Validierung {best['val_uar']:.1%})")
    print(f"Test: Trefferquote pro Gefühl gemittelt (UAR) {uar(y_test, prediction):.1%}\n")
    for i, r in zip(present, recall_score(y_test, prediction, labels=present, average=None)):
        print(f"  {EMOTIONS[i]:<8} {r:.1%}")

    print("\nVerwechslungsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    print(f"{'':<9}" + "".join(f"{EMOTIONS[i]:>9}" for i in present))
    for i, row in zip(present, confusion_matrix(y_test, prediction, labels=present)):
        print(f"{EMOTIONS[i]:<9}" + "".join(f"{n:>9}" for n in row))

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {"state": best["state"], "layer": best["layer"], "mean": best["mean"], "std": best["std"], "emotions": EMOTIONS},
        CHECKPOINT_DIR / "mlp_wav2vec.pt",
    )
    print("\nGespeichert: checkpoints/mlp_wav2vec.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cremad")
    main(parser.parse_args().dataset)
