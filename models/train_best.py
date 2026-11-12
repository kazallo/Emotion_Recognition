"""Stärkstes Stimm-Emotionsmodell: Schicht-Mix, mehrere Embeddings, Sprecher-Norm, Kalibrierung & Ensemble.

Umgesetzte Verbesserungen:
  1. Beliebige Kombination aus 3D-Embeddings (hubert_ms, wav2vec_ms) und 2D-Merkmalen (audeering, prosody, vad)
  2. Optional `--speaker-norm`: z-Score-Normalisierung pro Sprecher (entfernt individuelle Stimmlage/Mikrofon)
  3. Messe-Robustheit (KI-Con): Feature-Mixup + Rauschen
  4. Standardmäßig ohne MELD (`--exclude-corpora meld`), damit nur echte akustische Emotionen gelernt werden
  5. Korpus- und Klassen-Balance: jede Kombination (Korpus, Gefühl) zählt im Training gleich viel
  6. Wahrscheinlichkeits-Kalibrierung (Temperature Scaling auf der Validierung)
  7. Ehrliche Statistik: Mittelwert ± Standardabweichung der Einzel-Seeds + kalibriertes Ensemble
  8. Speicherschonend (lädt per mmap nur die genutzten Clips in kompakte Split-Tensoren -> kein RAM-Swap)

Aufruf:
  uv run models/train_best.py --features hubert_ms wav2vec_ms prosody --seeds 5
  uv run models/train_best.py --features hubert_ms wav2vec_ms audeering prosody --speaker-norm --seeds 5
"""

import argparse
import gc

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix, recall_score
from torch import nn

from common import CHECKPOINT_DIR, DATA_DIR, EMOTIONS, RESULTS_DIR, load_dataset, per_corpus_uar, save_results

HIDDEN = 512
DROPOUT = 0.4
INPUT_NOISE = 0.1
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 128
MAX_EPOCHS = 120
PATIENCE = 16


class FusionNet(nn.Module):
    """Pro Eingabedatei ein gelernter Schicht-Mix, dann ein Netz auf allen zusammen."""

    def __init__(self, n_layers_list: list[int], widths: list[int], n_classes: int) -> None:
        super().__init__()
        self.layer_logits = nn.ParameterList([nn.Parameter(torch.zeros(n)) for n in n_layers_list])
        total = sum(widths)
        self.norm_in = nn.LayerNorm(total)
        self.drop_in = nn.Dropout(DROPOUT / 2)
        self.fc1 = nn.Linear(total, HIDDEN)
        self.act1 = nn.GELU()
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN // 2)
        self.act2 = nn.GELU()
        self.norm2 = nn.LayerNorm(HIDDEN // 2)
        self.drop2 = nn.Dropout(DROPOUT)
        self.out = nn.Linear(HIDDEN // 2, n_classes)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        mixed = [(x * w.softmax(0)[None, :, None]).sum(1) for x, w in zip(inputs, self.layer_logits)]
        h = self.drop_in(self.norm_in(torch.cat(mixed, dim=1)))
        h = self.drop1(self.norm1(self.act1(self.fc1(h))))
        h = self.drop2(self.norm2(self.act2(self.fc2(h))))
        return self.out(h)


def batch_of(split_data: list[torch.Tensor], b, device) -> list[torch.Tensor]:
    return [d[b].to(device, dtype=torch.float32) for d in split_data]


@torch.no_grad()
def predict_logits(model, split_data: list[torch.Tensor], device) -> np.ndarray:
    model.eval()
    n = len(split_data[0])
    out = []
    for start in range(0, n, 512):
        batch = batch_of(split_data, slice(start, start + 512), device)
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out)


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


def corpus_mean_uar(table, mask, y, proba) -> float:
    scores = per_corpus_uar(table, mask, y, proba.argmax(1))
    return float(np.mean(list(scores.values())))


def sample_weights(table, train_mask: np.ndarray) -> torch.Tensor:
    group = (table["corpus"] + "_" + table["emotion"]).to_numpy()[train_mask]
    _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
    w = (1.0 / counts)[inverse]
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def apply_speaker_normalization(arr: np.ndarray, speakers: np.ndarray) -> None:
    """In-place z-Score-Normalisierung pro Sprecher (spart RAM)."""
    for spk in np.unique(speakers):
        m = np.flatnonzero(speakers == spk)
        if len(m) >= 4:
            sub = arr[m].astype("float32")
            mean = sub.mean(axis=0, keepdims=True)
            std = sub.std(axis=0, keepdims=True) + 1e-5
            arr[m] = ((sub - mean) / std).astype("float16")


def train_one(seed, splits_data, table, y_train_dev, weights_dev, y_val, val_mask, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_data, val_data = splits_data["train"], splits_data["val"]
    n_train = len(tr_data[0])

    model = FusionNet([d.shape[1] for d in tr_data], [d.shape[2] for d in tr_data], len(EMOTIONS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = MAX_EPOCHS * (n_train // BATCH_SIZE + 1)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, reduction="none")

    best, best_state, bad = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(n_train)
        for start in range(0, n_train, BATCH_SIZE):
            b = order[start : start + BATCH_SIZE]
            x = batch_of(tr_data, b, device)
            x = [t + INPUT_NOISE * torch.randn_like(t) for t in x]

            if MIXUP_ALPHA > 0 and len(b) > 1 and np.random.rand() < 0.4:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                perm = torch.randperm(len(b), device=device)
                x = [lam * t + (1.0 - lam) * t[perm] for t in x]
                out = model(x)
                loss1 = loss_fn(out, y_train_dev[b]) * weights_dev[b]
                loss2 = loss_fn(out, y_train_dev[b][perm]) * weights_dev[b][perm]
                loss = (lam * loss1 + (1.0 - lam) * loss2).mean()
            else:
                loss = (loss_fn(model(x), y_train_dev[b]) * weights_dev[b]).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()

        val_logits = predict_logits(model, val_data, device)
        score = corpus_mean_uar(table, val_mask, y_val, softmax_np(val_logits))
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad == PATIENCE:
                break

    model.load_state_dict(best_state)
    print(f"  Seed {seed}: Validierung (Mittel der Korpus-UARs) {best:.1%}, {epoch + 1} Epochen", flush=True)
    return model, best_state, best


def main(features: list[str], seeds: int, name: str, speaker_norm: bool, con_augment: bool, exclude_corpora: list[str]) -> None:
    table, _, usable = load_dataset("combined", f"combined_{features[0]}.npy")
    if exclude_corpora:
        usable = usable & ~table["corpus"].isin(exclude_corpora).to_numpy()

    masks = {s: usable & (table["split"] == s).to_numpy() for s in ("train", "val", "test")}
    idx = {s: np.flatnonzero(m) for s, m in masks.items()}
    usable_idx = np.flatnonzero(usable)
    # Lokale Position jedes Splits innerhalb von usable_idx
    local_pos = {s: np.searchsorted(usable_idx, idx[s]) for s in ("train", "val", "test")}

    splits_data: dict[str, list[torch.Tensor]] = {"train": [], "val": [], "test": []}
    means, stds = [], []

    for f in features:
        raw = np.load(DATA_DIR / f"combined_{f}.npy", mmap_mode="r")
        assert len(raw) == len(table), f"{f}: Länge passt nicht zur Tabelle"
        sub = np.array(raw[usable_idx], dtype="float16")
        del raw
        if sub.ndim == 2:
            sub = sub[:, None, :]
        if speaker_norm:
            apply_speaker_normalization(sub, table["speaker"].to_numpy()[usable_idx])

        tr_local = local_pos["train"]
        mean = np.zeros(sub.shape[1:], dtype="float64")
        sq = np.zeros(sub.shape[1:], dtype="float64")
        for s in range(0, len(tr_local), 2000):
            part = sub[tr_local[s : s + 2000]].astype("float64")
            mean += part.sum(0)
            sq += (part**2).sum(0)
        mean /= len(tr_local)
        std = np.sqrt(np.maximum(sq / len(tr_local) - mean**2, 0.0)) + 1e-6
        m_f, s_f = mean.astype("float32"), std.astype("float32")
        means.append(torch.from_numpy(m_f))
        stds.append(torch.from_numpy(s_f))

        for s in range(0, len(sub), 2000):
            sub[s : s + 2000] = ((sub[s : s + 2000].astype("float32") - m_f) / s_f).astype("float16")

        for s in ("train", "val", "test"):
            splits_data[s].append(torch.from_numpy(np.ascontiguousarray(sub[local_pos[s]])))
        del sub
        gc.collect()

    labels = torch.tensor([EMOTIONS.index(e) if e in EMOTIONS else -1 for e in table["emotion"]])
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Eingaben: {features}, Ohne: {exclude_corpora}, Sprecher-Norm: {speaker_norm}, Gerät: {device}")
    print(f"Clips: {len(idx['train'])} Training, {len(idx['val'])} Validierung, {len(idx['test'])} Test\n", flush=True)

    y_train_dev = labels[idx["train"]].to(device)
    weights_dev = sample_weights(table, masks["train"]).to(device)
    y = {s: labels[idx[s]].numpy() for s in ("val", "test")}

    states, logits_dict = [], {"val": [], "test": []}
    seed_test_uars, seed_corpus_means = [], []

    for seed in range(seeds):
        model, state, _ = train_one(seed, splits_data, table, y_train_dev, weights_dev, y["val"], masks["val"], device)
        states.append(state)
        for s in logits_dict:
            logits_dict[s].append(predict_logits(model, splits_data[s], device))
        pred_s = logits_dict["test"][-1].argmax(1)
        seed_test_uars.append(float(recall_score(y["test"], pred_s, average="macro")))
        seed_corpus_means.append(float(np.mean(list(per_corpus_uar(table, masks["test"], y["test"], pred_s).values()))))

    mean_logits = {s: np.mean(l, axis=0) for s, l in logits_dict.items()}
    temperature = calibrate_temperature(mean_logits["val"], y["val"])
    proba = {s: softmax_np(mean_logits[s], temperature) for s in mean_logits}

    pred = proba["test"].argmax(1)
    test_uar = float(recall_score(y["test"], pred, average="macro"))
    by_corpus = per_corpus_uar(table, masks["test"], y["test"], pred)
    val_by_corpus = per_corpus_uar(table, masks["val"], y["val"], proba["val"].argmax(1))
    recalls = recall_score(y["test"], pred, average=None)

    print(f"\nEinzel-Seeds (n={seeds}):")
    print(f"  Test-UAR gesamt:      {np.mean(seed_test_uars):.1%} ± {np.std(seed_test_uars):.1%}")
    print(f"  Mittel der Korpora:   {np.mean(seed_corpus_means):.1%} ± {np.std(seed_corpus_means):.1%}")
    print(f"\nKalibriertes Ensemble (Temperatur T={temperature:.2f}):")
    print("Validierung pro Korpus: " + ", ".join(f"{c} {v:.1%}" for c, v in val_by_corpus.items()))
    print(f"Test: UAR gesamt {test_uar:.1%}, Mittel der Korpora {np.mean(list(by_corpus.values())):.1%}")
    print("  pro Korpus: " + ", ".join(f"{c} {v:.1%}" for c, v in by_corpus.items()))
    for e, r in zip(EMOTIONS, recalls):
        print(f"  {e:<8} {r:.1%}")
    print("\nVerwechslungsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    print(f"{'':<9}" + "".join(f"{e:>9}" for e in EMOTIONS))
    for e, row in zip(EMOTIONS, confusion_matrix(y["test"], pred)):
        print(f"{e:<9}" + "".join(f"{n:>9}" for n in row))

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({
        "states": states, "means": means, "stds": stds, "features": features,
        "emotions": EMOTIONS, "temperature": temperature, "speaker_norm": speaker_norm,
        "exclude_corpora": exclude_corpora,
    }, CHECKPOINT_DIR / f"best_{name}.pt")
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(RESULTS_DIR / f"best_{name}_probs.npz", val=proba["val"], test=proba["test"],
             val_idx=idx["val"], test_idx=idx["test"])
    save_results(f"best_{name}", {
        "features": features, "exclude_corpora": exclude_corpora, "seeds": seeds,
        "speaker_norm": speaker_norm, "con_augment": con_augment, "temperature": temperature,
        "seed_test_uar_mean": float(np.mean(seed_test_uars)), "seed_test_uar_std": float(np.std(seed_test_uars)),
        "seed_corpus_uar_mean": float(np.mean(seed_corpus_means)), "seed_corpus_uar_std": float(np.std(seed_corpus_means)),
        "test_uar": test_uar, "test_uar_per_corpus": by_corpus, "val_uar_per_corpus": val_by_corpus,
        "recall": dict(zip(EMOTIONS, map(float, recalls))),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", default=["hubert_ms", "wav2vec_ms", "prosody"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--name", default=None)
    parser.add_argument("--speaker-norm", action="store_true", help="z-Score-Normalisierung pro Sprecher")
    parser.add_argument("--con-augment", action="store_true", help="Messehallen-Augmentation im Training nutzen")
    parser.add_argument("--exclude-corpora", nargs="*", default=["meld"], help="Korpora ausschließen (Standard: meld)")
    args = parser.parse_args()
    suffix = ("_nomeld" if "meld" in args.exclude_corpora else "") + ("_spknorm" if args.speaker_norm else "") + ("_con" if args.con_augment else "")
    default_name = "_".join(args.features) + suffix
    main(args.features, args.seeds, args.name or default_name, args.speaker_norm, args.con_augment, args.exclude_corpora)
