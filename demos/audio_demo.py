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
sys.path.append(str(ROOT / "models"))
sys.path.append(str(ROOT / "data_preprocessing"))
 
from audio import SAMPLE_RATE, load_audio  # noqa: E402
from common import CHECKPOINT_DIR  # noqa: E402
from train_mlp import EmotionMLP  # noqa: E402
 
# Diese Encoder müssen exakt die sein, mit denen du die Embeddings fürs Training extrahiert hast.
# Wenn du ein anderes Modell benutzt hast (z. B. DistilHuBERT über s3prl), hier anpassen.
ENCODERS = {
    "wav2vec": ("facebook/wav2vec2-large-robust", Wav2Vec2Model),
    "hubert": ("facebook/hubert-large-ll60k", HubertModel),
    "distilhubert": ("ntu-spl/distilhubert", HubertModel),
}
MIN_SECONDS, MAX_SECONDS = 0.5, 30.0
 
app = Flask(__name__)
device = "mps" if torch.backends.mps.is_available() else "cpu"
infer_lock = threading.Lock()  # eine Berechnung gleichzeitig (MPS und Speicher)
_encoders: dict = {}
_bundles: dict = {}
 
 
class Encoder:
    """Dieselbe Berechnung wie beim Extrahieren: pro Schicht der Mittelwert über die Zeit."""
 
    def __init__(self, name: str) -> None:
        model_name, model_class = ENCODERS[name]
        print(f"Lade Encoder {name} ({model_name}) auf {device} ...")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name).to(device).eval()
 
    @torch.inference_mode()
    def embed(self, wave: np.ndarray) -> np.ndarray:
        inputs = self.extractor(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        output = self.model(inputs.input_values.to(device), output_hidden_states=True)
        layers = torch.stack(output.hidden_states)  # (Schichten, 1, Zeit, Dim)
        return layers.mean(dim=2).squeeze(1).cpu().numpy()
 
 
def get_encoder(name: str) -> Encoder:
    if name not in _encoders:
        _encoders[name] = Encoder(name)
    return _encoders[name]
 
 
def encoder_of(stem: str):
    """mlp_hubert -> hubert, mlp_cremad_distilhubert -> distilhubert."""
    name = stem.split("_")[-1]
    return name if name in ENCODERS else None
 
 
def load_bundle(stem: str) -> dict:
    """Checkpoint und MLP laden (mit Zwischenspeicher)."""
    if stem not in _bundles:
        ckpt = torch.load(CHECKPOINT_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or not {"state", "layer", "mean", "std", "emotions"} <= ckpt.keys():
            raise ValueError("Checkpoint enthält nicht state, layer, mean, std und emotions (neu trainieren?)")
        model = EmotionMLP(len(np.asarray(ckpt["mean"])), len(ckpt["emotions"]))
        model.load_state_dict(ckpt["state"])
        model.eval()
        _bundles[stem] = {"ckpt": ckpt, "model": model, "encoder": encoder_of(stem)}
    return _bundles[stem]
 
 
@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")
 
 
def available_models() -> list[dict]:
    models = []
    for path in sorted(CHECKPOINT_DIR.glob("mlp_*.pt")):
        stem = path.stem
        if encoder_of(stem) is None:
            continue
        try:
            bundle = load_bundle(stem)
        except Exception as error:  # kaputte Checkpoints nicht anbieten
            print(f"Überspringe {path.name}: {error}")
            continue
        models.append(
            {
                "id": stem,
                "encoder": bundle["encoder"],
                "layer": int(bundle["ckpt"]["layer"]),
                "emotions": list(bundle["ckpt"]["emotions"]),
            }
        )
    return models
 
 
@app.get("/api/models")
def list_models():
    return jsonify(available_models())
 
 
@app.post("/api/classify")
def classify():
    stem = request.args.get("model", "")
    if not (CHECKPOINT_DIR / f"{stem}.pt").exists() or encoder_of(stem) is None:
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
            return jsonify(error=f"Aufnahme ist {seconds:.1f} s lang, erlaubt sind {MIN_SECONDS} bis {MAX_SECONDS:.0f} s"), 400
 
        start = time.perf_counter()
        with infer_lock:
            embeddings = get_encoder(bundle["encoder"]).embed(wave)  # (Schichten, Dim)
            ckpt, layer = bundle["ckpt"], bundle["ckpt"]["layer"]
            mean, std = np.asarray(ckpt["mean"]), np.asarray(ckpt["std"])
            if layer >= embeddings.shape[0] or embeddings.shape[1] != mean.shape[0]:
                return jsonify(error=f"Checkpoint passt nicht zum Encoder: Schicht {layer}, Encoder liefert {embeddings.shape}"), 400
            # Beim Training wurden die Embeddings als float16 gespeichert, das hier nachbilden
            x = embeddings[layer].astype("float16").astype("float32")
            x = torch.from_numpy((x - mean) / std).float().unsqueeze(0)
            with torch.no_grad():
                probs = bundle["model"](x).softmax(dim=1)[0].numpy()
        elapsed = time.perf_counter() - start
    except Exception as error:
        return jsonify(error=f"{type(error).__name__}: {error}"), 500
 
    emotions = list(bundle["ckpt"]["emotions"])
    return jsonify(
        probs={e: float(p) for e, p in zip(emotions, probs)},
        top=emotions[int(probs.argmax())],
        seconds=round(seconds, 2),
        compute_seconds=round(elapsed, 2),
    )
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", action="store_true", help="alle benötigten Encoder beim Start laden")
    args = parser.parse_args()
 
    if args.preload:
        for stem in (m["id"] for m in available_models()):
            get_encoder(encoder_of(stem))
    print(f"\nDemo läuft auf http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True)