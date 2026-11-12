"""MSP-Podcast-Emotions-Embeddings + Valenz/Arousal/Dominanz berechnen (audeering).

Warum dieses Modell?
  1. `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` wurde ausschließlich auf
     MSP-Podcast (über 100 Stunden natürliche englische Gespräche) feingetunt.
     Es hat KEINEN unserer Test-Datensätze (CREMA-D, RAVDESS, ESD, MELD) gesehen -> 0 % Data Leakage!
  2. Es liefert neben starken Emotions-Merkmalen direkt drei psychologische Dimensionen
     auf einer Skala von ~0 bis 1:
       - arousal   (Erregung: ruhig/gelangweilt vs. aufgeregt/wütend)
       - dominance (Dominanz: unsicher/unterwürfig vs. bestimmt/dominant)
       - valence   (Valenz: negativ/unangenehm vs. positiv/freundlich)
     Gerade die Stimm-Valenz ist für den späteren Ironie-Detektor Gold wert, weil man sie
     direkt von der Text-Valenz abziehen kann (positiver Text + negative Stimm-Valenz = Sarkasmus).
  3. Da es auf 12 Schichten gestutzt ist, rechnet es doppelt so schnell wie HuBERT-large.

Zusätzlich (für den Einsatz auf der KI-Con mit Mikrofon & Hintergrundgeräuschen):
  Mit `--augment-con` kann eine zweite Merkmalsdatei erzeugt werden, bei der auf die Clips
  leichter Raumhall + Messe-Hintergrundgeräusch (Babble/Pink Noise, 12-20 dB SNR) gelegt wird.

Ergebnis:
  - data/<datensatz>_audeering.npy: Form (Clips, 2051), float16
    (1024 Mittelwert + 1024 Standardabweichung der obersten Schicht + 3 [arousal, dominance, valence])
  - data/<datensatz>_vad.npy: Form (Clips, 3), float32 -> nur [arousal, dominance, valence]

Aufruf: uv run features/extract_audeering.py combined
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.append(str(ROOT / "data_preprocessing"))

from audio import SAMPLE_RATE, load_audio  # noqa: E402

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
CHUNK = 2000


class RegressionHead(nn.Module):
    """Der originale Regressionskopf von audeering für [arousal, dominance, valence]."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.dense(self.dropout(features)))
        return self.out_proj(self.dropout(x))


class EmotionModel(nn.Module):
    """Lädt exakt die veröffentlichten Gewichte (wav2vec2 + classifier) von audeering."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        state = load_file(hf_hub_download(model_name, "model.safetensors"))
        self.load_state_dict(state, strict=False)

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        last_hidden = outputs.last_hidden_state  # (1, Zeit, 1024)
        mean_pool = torch.mean(last_hidden, dim=1)
        std_pool = torch.std(last_hidden, dim=1, unbiased=False)
        vad = self.classifier(mean_pool)  # (1, 3): arousal, dominance, valence
        features = torch.cat([mean_pool, std_pool, vad], dim=-1)  # (1, 2051)
        return features.squeeze(0), vad.squeeze(0)


def simulate_con_environment(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simuliert eine Messehalle (KI-Con): leichter Raumhall + gedämpftes Hintergrund-Gemurmel."""
    if len(wave) < 400:
        return wave
    # 1. Pegel-Normalisierung (unabhängig vom Mikrofonabstand)
    peak = np.abs(wave).max()
    if peak > 1e-5:
        wave = wave / peak * rng.uniform(0.5, 0.95)

    # 2. Leichter Raumhall (frühe Reflexionen nach 15-45 ms)
    delay1 = int(SAMPLE_RATE * rng.uniform(0.015, 0.030))
    delay2 = int(SAMPLE_RATE * rng.uniform(0.035, 0.055))
    reverb = wave.copy()
    if len(wave) > delay2:
        reverb[delay1:] += 0.18 * wave[:-delay1]
        reverb[delay2:] += 0.09 * wave[:-delay2]

    # 3. Rosa/Babble-Rauschen (tieffrequentes Messe-Grundgeräusch, SNR 12 bis 22 dB)
    white = rng.standard_normal(len(reverb)).astype("float32")
    # Einfacher Tiefpass (gleitendes Mittel) für stimmähnliches Frequenzspektrum
    kernel = np.ones(15, dtype="float32") / 15.0
    babble = np.convolve(white, kernel, mode="same")
    sig_rms = np.sqrt(np.mean(reverb**2)) + 1e-6
    noise_rms = np.sqrt(np.mean(babble**2)) + 1e-6
    snr_db = rng.uniform(12.0, 22.0)
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noisy = reverb + babble * (target_noise_rms / noise_rms)
    return noisy.astype("float32")


@torch.inference_mode()
def embed_one(model: EmotionModel, processor, device: str, wave: np.ndarray) -> np.ndarray:
    if len(wave) < 400:
        wave = np.pad(wave, (0, 400 - len(wave)))
    inputs = processor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    feat, _ = model(inputs.input_values.to(device))
    return feat.float().cpu().numpy().astype("float16")


def main(dataset: str, augment_con: bool) -> None:
    table = pd.read_csv(DATA_DIR / f"{dataset}.csv")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = EmotionModel(MODEL_NAME).to(device).eval()

    suffix = "_audeering_con" if augment_con else "_audeering"
    print(f"{len(table)} Clips, {MODEL_NAME} ({suffix}), Gerät: {device}")

    part_dir = DATA_DIR / f"_{dataset}{suffix}_parts"
    part_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    for start in range(0, len(table), CHUNK):
        part = part_dir / f"{start:06d}.npy"
        if part.exists():
            continue
        sub = table.iloc[start : start + CHUNK]
        rows = []
        for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"ab {start}"):
            if row["corpus"] == "meld":
                rows.append(np.zeros(2051, dtype="float16"))
                continue
            w = load_audio(DATA_DIR / row["path"])
            if augment_con:
                w = simulate_con_environment(w, rng)
            rows.append(embed_one(model, processor, device, w))
        np.save(part, np.stack(rows))
        if device == "mps":
            torch.mps.empty_cache()

    embeddings = np.concatenate([np.load(p) for p in sorted(part_dir.glob("*.npy"))])
    assert len(embeddings) == len(table)
    np.save(DATA_DIR / f"{dataset}{suffix}.npy", embeddings)
    if not augment_con:
        vad = embeddings[:, -3:].astype("float32")
        np.save(DATA_DIR / f"{dataset}_vad.npy", vad)
        print(f"Gespeichert: data/{dataset}_vad.npy (Arousal, Dominance, Valence) mit Form {vad.shape}")

    for p in part_dir.glob("*.npy"):
        p.unlink()
    part_dir.rmdir()
    print(f"Gespeichert: data/{dataset}{suffix}.npy mit Form {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="combined")
    parser.add_argument("--augment-con", action="store_true", help="Messehallen-Simulation (Hall + Hintergrundgeräusch)")
    args = parser.parse_args()
    main(args.dataset, args.augment_con)
