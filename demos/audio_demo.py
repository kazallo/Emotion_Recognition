"""Web-Demo: Eigene Sprache im Browser aufnehmen und mit den trainierten Modellen testen.

Unterstützt sowohl die neuen Multi-Feature-Ensemble-Modelle (`best_*.pt`, inkl.
Sprecher-Normierung `_spknorm` und Prosodie) als auch die älteren Einzel-Encoder-Modelle (`mlp_*.pt`).

Für Modelle mit Sprecher-Normierung (`_spknorm`) sammelt der Server laufend die Aufnahmen
des aktuellen Sprechers in einem Sprecher-Profil und schätzt daraus Mittelwert und Streuung
der eigenen Stimme (mit Bayes-Glättung gegen die globale Trainingsverteilung, sodass schon
die 1. Aufnahme sinnvoll klassifiziert wird und jede weitere Aufnahme das Profil schärft).

Starten mit:  uv run demos/audio_demo.py
Dann im Browser http://127.0.0.1:8000 öffnen.
"""

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2Model

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
sys.path.extend([str(ROOT / "data_preprocessing"), str(ROOT / "models"), str(ROOT / "features")])

from audio import SAMPLE_RATE, load_audio  # noqa: E402
from extract_prosody import extract_prosody  # noqa: E402
from train_best import FusionNet, softmax_np  # noqa: E402
from train_mlp import EmotionMLP, LayerMixMLP  # noqa: E402

ENCODERS = {
    "hubert": ("facebook/hubert-large-ll60k", HubertModel),
    "wav2vec": ("facebook/wav2vec2-large-robust", Wav2Vec2Model),
}
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
# Bayes-Glättung (Shrinkage) für wenige Aufnahmen pro Sprecher:
# verhindert Division durch 0 bei der 1. Aufnahme und stabilisiert die Streuung bei 2-4 Aufnahmen
PRIOR_WEIGHT_MEAN = 1.0
PRIOR_WEIGHT_STD = 3.0


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AudioEncoder:
    def __init__(self, key: str) -> None:
        model_name, model_class = ENCODERS[key]
        self.device = pick_device()
        print(f"Lade {model_name} auf {self.device} …", flush=True)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def embed_mean_and_ms(self, wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Liefert (mean, mean+std) über die Zeit: (Schichten, 1024) und (Schichten, 2048) in float16."""
        inputs = self.extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        hidden = self.model(inputs.input_values.to(self.device), output_hidden_states=True).hidden_states
        layers = torch.stack(hidden).squeeze(1).float()  # (Schichten, Zeit, 1024)
        mean = layers.mean(dim=1)
        std = layers.std(dim=1, unbiased=False)
        ms = torch.cat([mean, std], dim=-1)
        return (
            mean.cpu().numpy().astype("float16"),
            ms.cpu().numpy().astype("float16"),
        )


app = Flask(__name__)
encoders: dict[str, AudioEncoder] = {}
bundles: dict[str, dict] = {}
# Gespeicherte Aufnahmen des aktuellen Sprechers: Liste aus Dicts {id, name, seconds, features: {feat: ndarray(L, D)}}
speaker_clips: list[dict] = []
next_clip_id = 1
infer_lock = threading.Lock()


def get_encoder(key: str) -> AudioEncoder:
    if key not in encoders:
        encoders[key] = AudioEncoder(key)
    return encoders[key]


def extract_needed_features(wave: np.ndarray, needed: set[str], cache: dict[str, np.ndarray]) -> None:
    """Berechnet alle noch fehlenden Merkmale für `wave` und legt sie mit Form (Schichten, Dim) in `cache` ab."""
    if ("hubert" in needed or "hubert_ms" in needed) and ("hubert_ms" not in cache):
        mean, ms = get_encoder("hubert").embed_mean_and_ms(wave)
        cache["hubert"] = mean
        cache["hubert_ms"] = ms
    if ("wav2vec" in needed or "wav2vec_ms" in needed) and ("wav2vec_ms" not in cache):
        mean, ms = get_encoder("wav2vec").embed_mean_and_ms(wave)
        cache["wav2vec"] = mean
        cache["wav2vec_ms"] = ms
    if "prosody" in needed and "prosody" not in cache:
        p = extract_prosody(wave).astype("float16")
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        cache["prosody"] = p[None, :]  # (1, 20)


def load_prior_stats(features: list[str], exclude_corpora: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Sucht das passende unnormierte Gegenstück-Checkpoint als Prior für die Sprecher-Normierung."""
    suffix = "_nomeld" if "meld" in exclude_corpora else ""
    candidate = CHECKPOINT_DIR / f"best_{'_'.join(features)}{suffix}.pt"
    if not candidate.exists():
        return None
    raw_ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
    if list(raw_ckpt.get("features", [])) != features:
        return None
    return (
        [np.asarray(m, dtype="float32") for m in raw_ckpt["means"]],
        [np.asarray(s, dtype="float32") for s in raw_ckpt["stds"]],
    )


class LegacyFusionNet(torch.nn.Module):
    """Ältere FusionNet-Variante (mit Sequenz-Head statt Zwischen-LayerNorms)."""

    def __init__(self, n_layers_list: list[int], widths: list[int], n_classes: int) -> None:
        super().__init__()
        self.layer_logits = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(n)) for n in n_layers_list])
        total = sum(widths)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(total),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(total, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, n_classes),
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        mixed = [(x * w.softmax(0)[None, :, None]).sum(1) for x, w in zip(inputs, self.layer_logits)]
        return self.head(torch.cat(mixed, dim=1))


def load_bundle(stem: str) -> dict:
    if stem in bundles:
        return bundles[stem]
    ckpt = torch.load(CHECKPOINT_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Kein gültiges Checkpoint-Dict")

    # 1. Neues Ensemble-Modell aus train_best.py (`best_*.pt`)
    if "states" in ckpt and "features" in ckpt:
        features = list(ckpt["features"])
        Unsupported = [f for f in features if f not in {"hubert_ms", "wav2vec_ms", "prosody"}]
        if Unsupported:
            raise ValueError(f"Merkmale in der Demo noch nicht unterstützt: {Unsupported}")
        means = [np.asarray(m, dtype="float32") for m in ckpt["means"]]
        stds = [np.asarray(s, dtype="float32") for s in ckpt["stds"]]
        n_layers = [m.shape[0] for m in means]
        widths = [m.shape[1] for m in means]
        emotions = list(ckpt["emotions"])
        models = []
        for state in ckpt["states"]:
            cls = LegacyFusionNet if "head.0.weight" in state else FusionNet
            m = cls(n_layers, widths, len(emotions))
            m.load_state_dict(state)
            m.eval()
            models.append(m)
        speaker_norm = bool(ckpt.get("speaker_norm", False))
        exclude_corpora = list(ckpt.get("exclude_corpora", []))
        prior = load_prior_stats(features, exclude_corpora) if speaker_norm else None
        bundles[stem] = {
            "kind": "best",
            "features": features,
            "models": models,
            "means": means,
            "stds": stds,
            "prior": prior,
            "temperature": float(ckpt.get("temperature", 1.0)),
            "speaker_norm": speaker_norm,
            "exclude_corpora": exclude_corpora,
            "emotions": emotions,
        }
        return bundles[stem]

    # 2. Ältere MLP-Modelle aus train_mlp.py (`mlp_*.pt`)
    encoder = next((k for k in ENCODERS if k in stem), None)
    state_dict = ckpt.get("state") or ckpt.get("state_dict")
    if encoder is None or state_dict is None:
        raise ValueError(f"Kein unterstütztes MLP-Checkpoint: {stem}")
    mode = ckpt.get("mode", "single")
    mean = np.asarray(ckpt["mean"], dtype="float32")
    std = np.asarray(ckpt["std"], dtype="float32")
    emotions = list(ckpt["emotions"])
    if mode == "mix":
        model = LayerMixMLP(mean.shape[0], mean.shape[1], len(emotions))
    else:
        model = EmotionMLP(mean.shape[0], len(emotions))
    model.load_state_dict(state_dict)
    model.eval()
    bundles[stem] = {
        "kind": "mlp",
        "mode": mode,
        "features": [encoder],
        "encoder": encoder,
        "layer": ckpt.get("layer", "mix"),
        "mean": mean,
        "std": std,
        "model": model,
        "speaker_norm": False,
        "exclude_corpora": [],
        "emotions": emotions,
    }
    return bundles[stem]


def predict_for_clip(bundle: dict, clip_features: dict[str, np.ndarray], all_clips: list[dict]) -> dict[str, float]:
    """Berechnet die Wahrscheinlichkeiten eines Clips (bei `speaker_norm=True` relativ zu `all_clips`)."""
    emotions = bundle["emotions"]
    if bundle["kind"] == "mlp":
        enc = bundle["encoder"]
        emb = clip_features[enc].astype("float32")
        if bundle["mode"] == "mix":
            x = (emb - bundle["mean"]) / bundle["std"]
        else:
            layer = int(bundle["layer"])
            x = (emb[layer] - bundle["mean"]) / bundle["std"]
        with torch.no_grad():
            probs = bundle["model"](torch.from_numpy(x).float().unsqueeze(0)).softmax(dim=1)[0].numpy()
        return {e: float(p) for e, p in zip(emotions, probs)}

    # `best_*.pt` Ensemble-Modell
    tensors = []
    n_spk = len(all_clips)
    for i, feat_name in enumerate(bundle["features"]):
        x = clip_features[feat_name].astype("float32")  # (L, D)
        if bundle["speaker_norm"]:
            stack = np.stack([c["features"][feat_name].astype("float32") for c in all_clips], axis=0)  # (N, L, D)
            spk_mean = stack.mean(axis=0)
            spk_std = stack.std(axis=0) if n_spk > 1 else np.zeros_like(spk_mean)
            if bundle["prior"] is not None:
                mu0, sd0 = bundle["prior"][0][i], bundle["prior"][1][i]
                mean_eff = (n_spk * spk_mean + PRIOR_WEIGHT_MEAN * mu0) / (n_spk + PRIOR_WEIGHT_MEAN)
                std_eff = ((n_spk - 1) * spk_std + PRIOR_WEIGHT_STD * sd0) / max(n_spk - 1 + PRIOR_WEIGHT_STD, 1e-5) + 1e-5
            else:
                mean_eff = spk_mean if n_spk > 1 else np.zeros_like(spk_mean)
                std_eff = (spk_std + 1e-5) if n_spk > 1 else np.ones_like(spk_mean)
            x = ((x - mean_eff) / std_eff).astype("float16").astype("float32")

        x = ((x - bundle["means"][i]) / bundle["stds"][i]).astype("float16").astype("float32")
        tensors.append(torch.from_numpy(x).unsqueeze(0))

    with torch.no_grad():
        logits = np.mean([m(tensors).numpy() for m in bundle["models"]], axis=0)
    probs = softmax_np(logits, bundle["temperature"])[0]
    return {e: float(p) for e, p in zip(emotions, probs)}


def available_models() -> list[dict]:
    models = []
    paths = sorted(CHECKPOINT_DIR.glob("best_*.pt")) + sorted(CHECKPOINT_DIR.glob("mlp_*.pt"))
    for path in paths:
        stem = path.stem
        if stem == "best_hubert_test" or "preprocessed" in stem:
            continue
        try:
            b = load_bundle(stem)
        except Exception as error:
            print(f"Überspringe {path.name}: {error}")
            continue
        models.append({
            "id": stem,
            "kind": b["kind"],
            "features": b["features"],
            "speaker_norm": b["speaker_norm"],
            "nomeld": "meld" in b["exclude_corpora"] or "_nomeld" in stem,
            "temperature": round(b.get("temperature", 1.0), 2),
            "emotions": b["emotions"],
        })
    # Empfohlenes Modell (ohne MELD + mit Sprecher-Normierung) nach ganz oben sortieren
    preferred = "best_hubert_ms_wav2vec_ms_prosody_nomeld_spknorm"
    models.sort(key=lambda m: (0 if m["id"] == preferred else (1 if m["id"].startswith("best_") else 2), m["id"]))
    return models


def evaluate_history(bundle: dict) -> list[dict]:
    """Berechnet für alle bisher gespeicherten Aufnahmen im Sprecher-Profil die aktuelle Vorhersage."""
    history = []
    for clip in speaker_clips:
        probs = predict_for_clip(bundle, clip["features"], speaker_clips)
        top = max(probs, key=probs.get)
        history.append({
            "id": clip["id"],
            "name": clip["name"],
            "seconds": clip["seconds"],
            "probs": probs,
            "top": top,
        })
    return history


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.get("/api/models")
def list_models():
    return jsonify(models=available_models(), speaker_clips=len(speaker_clips))


@app.post("/api/speaker/reset")
def reset_speaker():
    with infer_lock:
        speaker_clips.clear()
    return jsonify(speaker_clips=0, history=[])


@app.post("/api/speaker/delete/<int:clip_id>")
def delete_clip(clip_id: int):
    stem = request.args.get("model", "")
    with infer_lock:
        speaker_clips[:] = [c for c in speaker_clips if c["id"] != clip_id]
        history = []
        if stem and (CHECKPOINT_DIR / f"{stem}.pt").exists():
            bundle = load_bundle(stem)
            for c in speaker_clips:
                extract_needed_features(c["wave"], set(bundle["features"]), c["features"])
            history = evaluate_history(bundle)
    return jsonify(speaker_clips=len(speaker_clips), history=history)


@app.post("/api/reclassify")
def reclassify():
    """Bewertet alle vorhandenen Aufnahmen im Sprecher-Profil mit einem neu gewählten Modell."""
    stem = request.args.get("model", "")
    if not (CHECKPOINT_DIR / f"{stem}.pt").exists():
        return jsonify(error=f"Unbekanntes Modell: {stem!r}"), 400
    try:
        start = time.perf_counter()
        with infer_lock:
            bundle = load_bundle(stem)
            needed = set(bundle["features"])
            for clip in speaker_clips:
                extract_needed_features(clip["wave"], needed, clip["features"])
            history = evaluate_history(bundle)
        elapsed = time.perf_counter() - start
    except Exception as error:
        return jsonify(error=f"{type(error).__name__}: {error}"), 500

    if not history:
        return jsonify(speaker_clips=0, history=[], compute_seconds=round(elapsed, 2))
    latest = history[-1]
    return jsonify(
        probs=latest["probs"],
        top=latest["top"],
        seconds=latest["seconds"],
        compute_seconds=round(elapsed, 2),
        speaker_clips=len(speaker_clips),
        history=history,
    )


@app.post("/api/classify")
def classify():
    global next_clip_id
    stem = request.args.get("model", "")
    label = request.args.get("name", "").strip()
    if not (CHECKPOINT_DIR / f"{stem}.pt").exists():
        return jsonify(error=f"Unbekanntes Modell: {stem!r}"), 400
    body = request.get_data()
    if not body:
        return jsonify(error="Keine Audiodaten empfangen"), 400

    try:
        bundle = load_bundle(stem)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(body)
            tmp.flush()
            wave = load_audio(Path(tmp.name))
        seconds = len(wave) / SAMPLE_RATE
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            return jsonify(
                error=f"Aufnahme ist {seconds:.1f} s lang, erlaubt sind {MIN_SECONDS} bis {MAX_SECONDS:.0f} s"
            ), 400

        start = time.perf_counter()
        with infer_lock:
            # Alle Merkmale für HuBERT + Wav2Vec + Prosodie direkt berechnen, damit Modellwechsel sofort gehen
            needed = set(bundle["features"]) | {"hubert_ms", "wav2vec_ms", "prosody"}
            feat_cache: dict[str, np.ndarray] = {}
            extract_needed_features(wave, needed, feat_cache)
            clip_entry = {
                "id": next_clip_id,
                "name": label or f"Aussage #{next_clip_id}",
                "seconds": round(seconds, 2),
                "wave": wave,
                "features": feat_cache,
            }
            next_clip_id += 1
            speaker_clips.append(clip_entry)
            history = evaluate_history(bundle)
        elapsed = time.perf_counter() - start
    except Exception as error:
        return jsonify(error=f"{type(error).__name__}: {error}"), 500

    latest = history[-1]
    return jsonify(
        probs=latest["probs"],
        top=latest["top"],
        seconds=latest["seconds"],
        compute_seconds=round(elapsed, 2),
        speaker_clips=len(speaker_clips),
        history=history,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", action="store_true", help="HuBERT und Wav2Vec2 direkt beim Start laden")
    args = parser.parse_args()

    if args.preload:
        for key in ENCODERS:
            get_encoder(key)
    print(f"\nDemo läuft auf http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True)
