"""Sauberes Text-Emotions- & Valenz-Modell auf data/text_combined.csv (4 Korpora, roberta-base) trainieren.

Umgesetzte Verbesserungen:
  1. Neutrales Basis-Sprachmodell (`roberta-base`, 1536 Merkmale) ohne MELD-Vortraining & ohne Data Leakage
  2. Trainiert auf 4 reinen Text-Korpora (`carer`, `dailydialog`, `emocontext`, `goemotions`)
  3. Multi-Task-Ausgabe:
       - 4 Emotions-Klassen (`anger`, `joy`, `neutral`, `sadness`)
       - kontinuierliche Text-Valenz (`-1.0` negativ bis `+1.0` positiv) als direkter Gegenspieler zur Stimm-Valenz
  4. Temperatur-Kalibrierung (`calibrate_temperature`) auf der Validierung
  5. Ehrliche Statistik: Mittelwert ± Standardabweichung über alle Seeds + kalibriertes Ensemble

Ergebnis: checkpoints/text_emotion.pt und results/text_emotion.json

Aufruf: uv run models/train_text.py --seeds 5
"""

import argparse

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix, mean_absolute_error, recall_score
from torch import nn

from common import CHECKPOINT_DIR, EMOTIONS, load_dataset, per_corpus_uar, save_results

HIDDEN = 384
DROPOUT = 0.35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 10


class TextEmotionValenceNet(nn.Module):
    """Gemeinsames Rumpf-Netz mit zwei Köpfen: 4 Emotionsklassen + 1 Valenz-Score (-1..+1)."""

    def __init__(self, n_inputs: int, n_classes: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(n_inputs),
            nn.Dropout(0.15),
            nn.Linear(n_inputs, HIDDEN),
            nn.GELU(),
            nn.LayerNorm(HIDDEN),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.GELU(),
            nn.LayerNorm(HIDDEN // 2),
            nn.Dropout(DROPOUT),
        )
        self.emotion_head = nn.Linear(HIDDEN // 2, n_classes)
        self.valence_head = nn.Linear(HIDDEN // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.emotion_head(h)
        valence = torch.tanh(self.valence_head(h)).squeeze(-1)
        return logits, valence


@torch.no_grad()
def predict_all(model: nn.Module, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list, val_list = [], []
    for start in range(0, len(x), 1024):
        logits, val = model(x[start : start + 1024])
        logits_list.append(logits.cpu().numpy())
        val_list.append(val.cpu().numpy())
    return np.concatenate(logits_list), np.concatenate(val_list)


def softmax_np(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    scaled = logits / max(temp, 1e-3)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def calibrate_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    def nll(t: float) -> float:
        probs = np.clip(softmax_np(logits_val, t), 1e-7, 1.0)
        return float(-np.mean(np.log(probs[np.arange(len(y_val)), y_val])))

    res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
    return float(res.x)


def sample_weights(table, train_mask: np.ndarray) -> torch.Tensor:
    """Jede Kombination (Korpus, Gefühl) zählt im Training gleich stark."""
    group = (table["corpus"] + "_" + table["emotion"]).to_numpy()[train_mask]
    _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
    w = (1.0 / counts)[inverse]
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main(dataset: str, seeds: int) -> None:
    table, features, usable = load_dataset(dataset, f"{dataset}_roberta_clean.npy")
    features = features.astype("float32")
    labels = torch.tensor([EMOTIONS.index(e) if e in EMOTIONS else -1 for e in table["emotion"]])
    valences = torch.tensor(table["valence"].to_numpy(dtype="float32"))
    masks = {s: usable & (table["split"] == s).to_numpy() for s in ("train", "val", "test")}

    mean = features[masks["train"]].mean(axis=0)
    std = features[masks["train"]].std(axis=0) + 1e-6
    x_norm = torch.from_numpy((features - mean) / std)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Datensatz: {dataset} (roberta-base, 4 Korpora), Gerät: {device}")
    print(f"Sätze: {masks['train'].sum()} Training, {masks['val'].sum()} Validierung, {masks['test'].sum()} Test\n")

    x_tr = x_norm[masks["train"]].to(device)
    y_tr = labels[masks["train"]].to(device)
    v_tr = valences[masks["train"]].to(device)
    w_tr = sample_weights(table, masks["train"]).to(device)

    x_val = x_norm[masks["val"]].to(device)
    y_val = labels[masks["val"]].numpy()
    x_te = x_norm[masks["test"]].to(device)
    y_te = labels[masks["test"]].numpy()
    v_te = valences[masks["test"]].numpy()

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
    mse_loss = nn.SmoothL1Loss(reduction="none")

    states, val_logits_list, test_logits_list, test_val_list = [], [], [], []
    seed_test_uars, seed_corpus_means = [], []

    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = TextEmotionValenceNet(features.shape[1], len(EMOTIONS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_score, best_state, bad = -1.0, None, 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            order = torch.randperm(len(x_tr), device=device)
            for start in range(0, len(order), BATCH_SIZE):
                b = order[start : start + BATCH_SIZE]
                noisy = x_tr[b] + 0.05 * torch.randn_like(x_tr[b])
                logits, pred_v = model(noisy)
                loss = ((ce_loss(logits, y_tr[b]) + 0.3 * mse_loss(pred_v, v_tr[b])) * w_tr[b]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            logits_v, _ = predict_all(model, x_val)
            score = float(np.mean(list(per_corpus_uar(table, masks["val"], y_val, logits_v.argmax(1)).values())))
            if score > best_score:
                best_score, bad = score, 0
                best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad == PATIENCE:
                    break

        model.load_state_dict(best_state)
        print(f"  Seed {seed}: Validierung (Mittel der Korpus-UARs) {best_score:.1%}, {epoch + 1} Epochen", flush=True)
        states.append(best_state)
        lv, _ = predict_all(model, x_val)
        lt, vt = predict_all(model, x_te)
        val_logits_list.append(lv)
        test_logits_list.append(lt)
        test_val_list.append(vt)
        pred_s = lt.argmax(1)
        seed_test_uars.append(float(recall_score(y_te, pred_s, average="macro")))
        seed_corpus_means.append(float(np.mean(list(per_corpus_uar(table, masks["test"], y_te, pred_s).values()))))

    mean_val_logits = np.mean(val_logits_list, axis=0)
    mean_test_logits = np.mean(test_logits_list, axis=0)
    mean_test_valence = np.mean(test_val_list, axis=0)

    temperature = calibrate_temperature(mean_val_logits, y_val)
    proba_test = softmax_np(mean_test_logits, temperature)
    pred = proba_test.argmax(1)

    test_uar = float(recall_score(y_te, pred, average="macro"))
    by_corpus = per_corpus_uar(table, masks["test"], y_te, pred)
    recalls = recall_score(y_te, pred, average=None)
    valence_mae = float(mean_absolute_error(v_te, mean_test_valence))

    print(f"\nEinzel-Seeds (n={seeds}):")
    print(f"  Test-UAR gesamt:      {np.mean(seed_test_uars):.1%} ± {np.std(seed_test_uars):.1%}")
    print(f"  Mittel der Korpora:   {np.mean(seed_corpus_means):.1%} ± {np.std(seed_corpus_means):.1%}")
    print(f"\nKalibriertes Ensemble (Temperatur T={temperature:.2f}):")
    print(f"Test: UAR gesamt {test_uar:.1%}, Mittel der Korpora {np.mean(list(by_corpus.values())):.1%}")
    print(f"Valenz-MAE (Skala -1..+1): {valence_mae:.3f}")
    print("  pro Korpus: " + ", ".join(f"{c} {v:.1%}" for c, v in by_corpus.items()))
    for e, r in zip(EMOTIONS, recalls):
        print(f"  {e:<8} {r:.1%}")

    print("\nVerwechslungsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    print(f"{'':<9}" + "".join(f"{e:>9}" for e in EMOTIONS))
    for e, row in zip(EMOTIONS, confusion_matrix(y_te, pred)):
        print(f"{e:<9}" + "".join(f"{n:>9}" for n in row))

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({
        "states": states, "mean": mean, "std": std, "emotions": EMOTIONS,
        "temperature": temperature, "model_name": "roberta-base",
    }, CHECKPOINT_DIR / "text_emotion.pt")
    save_results("text_emotion", {
        "dataset": dataset, "backbone": "roberta-base", "seeds": seeds, "temperature": temperature,
        "seed_test_uar_mean": float(np.mean(seed_test_uars)), "seed_test_uar_std": float(np.std(seed_test_uars)),
        "seed_corpus_uar_mean": float(np.mean(seed_corpus_means)), "seed_corpus_uar_std": float(np.std(seed_corpus_means)),
        "test_uar": test_uar, "test_uar_per_corpus": by_corpus, "valence_mae": valence_mae,
        "recall": dict(zip(EMOTIONS, map(float, recalls))),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="text_combined")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    main(args.dataset, args.seeds)
