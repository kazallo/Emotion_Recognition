import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, HubertModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.append(str(ROOT / "data_preprocessing"))

from audio import SAMPLE_RATE, load_audio  # noqa: E402

# Die "robuste" Variante wurde auch mit Telefongesprächen und verrauschten
# Aufnahmen trainiert und kommt mit fremden Mikrofonen besser klar.
MODEL_NAME = "facebook/hubert-large-ll60k"


class HUBERTEncoder:
    def __init__(self) -> None:
        # Auf dem Mac rechnet die Grafikeinheit (MPS) deutlich schneller als der Prozessor
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        # eval(): Trainingseffekte wie Dropout aus. Das Modell wird nie verändert.
        self.model = HubertModel.from_pretrained(MODEL_NAME).to(self.device).eval()

    @torch.inference_mode()
    def embed(self, wave: np.ndarray) -> np.ndarray:
        inputs = self.extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        output = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        layers = torch.stack(output.hidden_states)  # (Schichten, 1, Zeit, 1024)
        return layers.mean(dim=2).squeeze(1).cpu().numpy()


def main(dataset: str) -> None:
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    encoder = HUBERTEncoder()
    print(f"{len(table)} Clips, Berechnung auf: {encoder.device}")

    # float32 
    embeddings = np.stack(
        [encoder.embed(load_audio(DATA_DIR / path)).astype("float32") for path in tqdm(table["path"], desc="Embeddings")]
    )

    np.save(DATA_DIR / f"{dataset}_hubert.npy", embeddings)
    print(f"Gespeichert: data/{dataset}_hubert.npy mit Form {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="combined")
    main(parser.parse_args().dataset)
