"""Gestapelter Autoencoder (35-15-15) wie bei Aouani & Ben Ayed (2020).

Der Autoencoder lernt, die 42 Merkmale durch einen Engpass von 15 Zahlen zu
pressen und daraus wieder herzustellen. Danach wird nur der vordere Teil
(Encoder) behalten: Seine 15 Zahlen pro Clip sind die neuen Merkmale für die SVM.

Gelernt wird nur mit dem Trainingsteil. Der Validierungsteil entscheidet,
wann das Training aufhört. Das Gefühl spielt beim Training keine Rolle.

Ergebnis:
  data/<datensatz>_encoded.npy   15 Zahlen pro Clip (gleiche Reihenfolge wie die Tabelle)
  checkpoints/autoencoder.pt     der trainierte Autoencoder samt Normalisierung

Aufruf: uv run models/autoencoder.py cremad
"""

import argparse

import numpy as np
import torch
from torch import nn

from common import CHECKPOINT_DIR, DATA_DIR, SEED, load_dataset

# Werte aus dem Paper
HIDDEN = [35, 15, 15]
WEIGHT_DECAY = 1e-5

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 1000
# Training endet, wenn der Fehler auf der Validierung so viele Epochen lang nicht sinkt
PATIENCE = 30


class StackedAutoencoder(nn.Module):
    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        # Encoder: 42 -> 35 -> 15 -> 15
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN[0]), nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ReLU(),
            nn.Linear(HIDDEN[1], HIDDEN[2]),
        )
        # Decoder: das Spiegelbild, 15 -> 15 -> 35 -> 42
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN[2], HIDDEN[1]), nn.ReLU(),
            nn.Linear(HIDDEN[1], HIDDEN[0]), nn.ReLU(),
            nn.Linear(HIDDEN[0], n_inputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def main(dataset: str) -> None:
    torch.manual_seed(SEED)
    table, features, usable = load_dataset(dataset, f"{dataset}_features.npy")
    train = usable & (table["split"] == "train").to_numpy()
    val = usable & (table["split"] == "val").to_numpy()

    # Normalisieren: Mittelwert 0 und Streuung 1, Werte nur aus dem Trainingsteil
    mean, std = features[train].mean(axis=0), features[train].std(axis=0) + 1e-8
    x = torch.from_numpy((features - mean) / std).float()
    x_train, x_val = x[train], x[val]
    print(f"Training mit {train.sum()} Clips, Validierung mit {val.sum()} Clips")

    model = StackedAutoencoder(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_loss, best_state, epochs_without_gain = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = x_train[order[start : start + BATCH_SIZE]]
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), x_val).item()
        if val_loss < best_loss:
            best_loss, epochs_without_gain = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain == PATIENCE:
                break
        if epoch % 50 == 0:
            print(f"Epoche {epoch}: Rekonstruktionsfehler Validierung {val_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"Fertig nach {epoch} Epochen, bester Rekonstruktionsfehler {best_loss:.4f}")

    # Alle Clips verdichten, auch die gerade nicht benutzten, damit die Reihenfolge zur Tabelle passt.
    # Clips mit fehlenden Werten bekommen dabei automatisch NaN.
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(x).numpy()
    np.save(DATA_DIR / f"{dataset}_encoded.npy", encoded)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "mean": mean, "std": std, "n_inputs": features.shape[1]},
        CHECKPOINT_DIR / "autoencoder.pt",
    )
    print(f"Gespeichert: data/{dataset}_encoded.npy mit Form {encoded.shape} und checkpoints/autoencoder.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cremad")
    main(parser.parse_args().dataset)