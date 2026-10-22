"""wav2vec-2.0-Embeddings für jeden Clip berechnen.

wav2vec 2.0 ist ein großes, vortrainiertes Sprachmodell. Wir benutzen es
eingefroren als "Ohr": Es macht aus jedem Clip pro Schicht einen Vektor mit
1024 Zahlen. Trainiert wird später nur ein kleines Netz auf diesen Vektoren.

Weil sich das eingefrorene Modell nie ändert, wird das nur einmal berechnet.

Ergebnis: data/<datensatz>_wav2vec.npy mit der Form (Clips, 25 Schichten, 1024),
in derselben Reihenfolge wie die Zeilen in data/<datensatz>.csv

Aufruf: uv run features/extract_wav2vec.py cremad
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2Model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.append(str(ROOT / "data_preprocessing"))

from audio import SAMPLE_RATE, load_audio  # noqa: E402

# Die "robuste" Variante wurde auch mit Telefongesprächen und verrauschten
# Aufnahmen trainiert und kommt mit fremden Mikrofonen besser klar.
MODEL_NAME = "facebook/wav2vec2-large-robust"


class Wav2VecEncoder:
    def __init__(self) -> None:
        # Auf dem Mac rechnet die Grafikeinheit (MPS) deutlich schneller als der Prozessor
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        # eval(): Trainingseffekte wie Dropout aus. Das Modell wird nie verändert.
        self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(self.device).eval()

    @torch.inference_mode()
    def embed(self, wave: np.ndarray) -> np.ndarray:
        """Gibt (Schichten, 1024) zurück: pro Schicht der Mittelwert über die Zeit.

        wav2vec liefert etwa 50 Vektoren pro Sekunde. Durch den Mittelwert hat jeder
        Clip gleich viele Zahlen, egal wie lang er ist. Wir speichern alle Schichten,
        weil sich Gefühle oft in den mittleren Schichten am besten ablesen lassen.
        """
        inputs = self.extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        output = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        layers = torch.stack(output.hidden_states)  # (Schichten, 1, Zeit, 1024)
        return layers.mean(dim=2).squeeze(1).cpu().numpy()


def main(dataset: str) -> None:
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    encoder = Wav2VecEncoder()
    print(f"{len(table)} Clips, Berechnung auf: {encoder.device}")

    # float16 reicht für die Embeddings und halbiert den Speicherplatz
    embeddings = np.stack(
        [encoder.embed(load_audio(DATA_DIR / path)).astype("float16") for path in tqdm(table["path"], desc="Embeddings")]
    )

    np.save(DATA_DIR / f"{dataset}_wav2vec.npy", embeddings)
    print(f"Gespeichert: data/{dataset}_wav2vec.npy mit Form {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="cremad")
    main(parser.parse_args().dataset)
