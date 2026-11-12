"""Text-Embeddings mit einem neutralen Basis-Sprachmodell (roberta-base) berechnen.

Warum `roberta-base` statt `j-hartmann/emotion-english-distilroberta-base`?
  1. `j-hartmann` wurde u. a. auf MELD, GoEmotions und CARER trainiert -> indirekte MELD-Kontamination
     und Überschneidung mit den Testdaten (Data Leakage).
  2. `roberta-base` ist ein komplett neutrales englisches Sprachmodell ohne Emotions- oder MELD-Vortraining.
  3. Wir extrahieren pro Satz sowohl den CLS-Vektor der obersten Schicht (768) als auch das
     aufmerksamkeits-gewichtete Mittel (Mean-Pooling) über die letzten 4 Transformer-Schichten (768)
     -> 1536 Merkmale pro Satz.

Ergebnis: data/<datensatz>_roberta_clean.npy mit Form (Sätze, 1536), float16.

Aufruf: uv run features/extract_text.py text_combined
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MODEL_NAME = "roberta-base"
BATCH_SIZE = 128


@torch.inference_mode()
def encode_texts(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """Gibt pro Satz 1536 Werte (768 CLS + 768 Mean-Pool der letzten 4 Schichten) als float16 zurück."""
    parts = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="roberta-base Embeddings"):
        batch = texts[start : start + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        cls_vec = out.hidden_states[-1][:, 0, :].float()
        # Mittelwert der letzten 4 Schichten, gemittelt über alle Nicht-Padding-Tokens
        last4 = torch.stack(out.hidden_states[-4:]).mean(dim=0).float()  # (B, T, 768)
        mask = enc.attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        mean_pool = (last4 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        vec = torch.cat([cls_vec, mean_pool], dim=1).cpu().numpy().astype("float16")
        parts.append(vec)
        if device == "mps" and (start // BATCH_SIZE) % 50 == 0:
            torch.mps.empty_cache()
    return np.concatenate(parts, axis=0)


def main(dataset: str) -> None:
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    texts = table["text"].fillna("").astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{len(texts)} Sätze aus {dataset}.csv, Modell: {MODEL_NAME}, Gerät: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    embeddings = encode_texts(texts, tokenizer, model, device)
    np.save(DATA_DIR / f"{dataset}_roberta_clean.npy", embeddings)
    print(f"Gespeichert: data/{dataset}_roberta_clean.npy mit Form {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="text_combined")
    main(parser.parse_args().dataset)
