"""Embeddings mit Mittelwert UND Standardabweichung über die Zeit berechnen.

Der reine Mittelwert (extract_wav2vec.py / extract_DistilHUBERT.py) wirft die Dynamik
weg: wie stark Tonhöhe und Betonung schwanken. Genau die ist für Gefühle und später
für Ironie wichtig. Hier werden beide Statistiken gespeichert.

Ergebnis: data/<datensatz>_<modell>_ms.npy mit Form (Clips, Schichten, 2048):
          erste 1024 Zahlen = Mittelwert, zweite 1024 = Standardabweichung.
Die Berechnung wird alle 1000 Clips zwischengespeichert und kann fortgesetzt werden.

Aufruf: uv run features/extract_stats.py combined --model hubert
        uv run features/extract_stats.py combined --model wav2vec
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2Model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.append(str(ROOT / "data_preprocessing"))

from audio import SAMPLE_RATE, load_audio  # noqa: E402

MODELS = {
    "hubert": ("facebook/hubert-large-ll60k", HubertModel),
    "wav2vec": ("facebook/wav2vec2-large-robust", Wav2Vec2Model),
}
CHUNK = 1000


@torch.inference_mode()
def embed(model, extractor, device, wave: np.ndarray) -> np.ndarray:
    inputs = extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    hidden = model(inputs.input_values.to(device), output_hidden_states=True).hidden_states
    layers = torch.stack(hidden).squeeze(1).float()  # (Schichten, Zeit, 1024)
    stats = torch.cat([layers.mean(dim=1), layers.std(dim=1, unbiased=False)], dim=-1)
    return stats.cpu().numpy().astype("float16")


def main(dataset: str, name: str) -> None:
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    model_name, model_class = MODELS[name]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name).to(device).eval()
    print(f"{len(table)} Clips, {name}, Berechnung auf: {device}")

    part_dir = DATA_DIR / f"_{dataset}_{name}_ms_parts"
    part_dir.mkdir(exist_ok=True)
    for start in range(0, len(table), CHUNK):
        part = part_dir / f"{start:06d}.npy"
        if part.exists():
            continue
        paths = table["path"].iloc[start : start + CHUNK]
        chunk = np.stack([embed(model, extractor, device, load_audio(DATA_DIR / p)) for p in tqdm(paths, desc=f"ab {start}")])
        np.save(part, chunk)
        if device == "mps":
            torch.mps.empty_cache()

    embeddings = np.concatenate([np.load(p) for p in sorted(part_dir.glob("*.npy"))])
    assert len(embeddings) == len(table)
    np.save(DATA_DIR / f"{dataset}_{name}_ms.npy", embeddings)
    for p in part_dir.glob("*.npy"):
        p.unlink()
    part_dir.rmdir()
    print(f"Gespeichert: data/{dataset}_{name}_ms.npy mit Form {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="combined")
    parser.add_argument("--model", default="hubert", choices=list(MODELS))
    args = parser.parse_args()
    main(args.dataset, args.model)
