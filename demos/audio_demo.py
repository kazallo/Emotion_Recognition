"""Web-Demo: Eigene Aussagen im Browser aufnehmen und Stimm-Modell + Text-Modell + Diskrepanz testen.

Funktionen:
  1. Stimm-Emotionsmodell (`best_*.pt`, Standard: 4 Merkmale ohne MELD mit Sprecher-Normierung, 90,8 % UAR)
  2. Live-Sprecher-Normierung über die bisherigen Aufnahmen des Sprechers (mit Bayes-Glättung ab Aufnahme 1)
  3. Automatische englische Spracherkennung (`openai/whisper-base.en`), falls kein Text eingetippt wurde
  4. Neues reines Text-Emotions- & Valenz-Modell (`checkpoints/text_emotion.pt` auf `roberta-base` ohne MELD)
  5. Diskrepanz-Analyse (Stimm-Emotion & Stimm-Valenz vs. wörtliche Text-Emotion & Text-Valenz als Ironie-Signal)

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
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HubertModel,
    Wav2Vec2Model,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
sys.path.extend([str(ROOT / "data_preprocessing"), str(ROOT / "models"), str(ROOT / "features")])

from audio import SAMPLE_RATE, load_audio  # noqa: E402
from extract_audeering import MODEL_NAME as AUDEERING_NAME, EmotionModel, embed_one as embed_audeering  # noqa: E402
from extract_prosody import extract_prosody  # noqa: E402
from train_best import FusionNet, softmax_np  # noqa: E402
from train_mlp import EmotionMLP, LayerMixMLP  # noqa: E402
from train_text import TextEmotionValenceNet  # noqa: E402

ENCODERS = {
    "hubert": ("facebook/hubert-large-ll60k", HubertModel),
    "wav2vec": ("facebook/wav2vec2-large-robust", Wav2Vec2Model),
}
WHISPER_NAME = "openai/whisper-base.en"
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
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
        layers = torch.stack(hidden).squeeze(1).float()
        mean = layers.mean(dim=1)
        std = layers.std(dim=1, unbiased=False)
        ms = torch.cat([mean, std], dim=-1)
        return (
            mean.cpu().numpy().astype("float16"),
            ms.cpu().numpy().astype("float16"),
        )


app = Flask(__name__)
encoders: dict[str, AudioEncoder] = {}
audeering_pair: tuple[EmotionModel, object, str] | None = None
whisper_pair: tuple[WhisperForConditionalGeneration, WhisperProcessor, str] | None = None
text_bundle: dict | None = None
bundles: dict[str, dict] = {}
speaker_clips: list[dict] = []
next_clip_id = 1
infer_lock = threading.Lock()


def get_encoder(key: str) -> AudioEncoder:
    if key not in encoders:
        encoders[key] = AudioEncoder(key)
    return encoders[key]


def get_audeering():
    global audeering_pair
    if audeering_pair is None:
        dev = pick_device()
        print(f"Lade {AUDEERING_NAME} auf {dev} …", flush=True)
        proc = AutoFeatureExtractor.from_pretrained(AUDEERING_NAME)
        mod = EmotionModel(AUDEERING_NAME).to(dev).eval()
        audeering_pair = (mod, proc, dev)
    return audeering_pair


def get_whisper():
    global whisper_pair
    if whisper_pair is None:
        dev = pick_device()
        print(f"Lade Spracherkennung {WHISPER_NAME} auf {dev} …", flush=True)
        proc = WhisperProcessor.from_pretrained(WHISPER_NAME)
        mod = WhisperForConditionalGeneration.from_pretrained(WHISPER_NAME).to(dev).eval()
        whisper_pair = (mod, proc, dev)
    return whisper_pair


@torch.inference_mode()
def transcribe_english(wave: np.ndarray) -> str:
    """Erkennt den gesprochenen englischen Satz aus dem Audio-Signal."""
    mod, proc, dev = get_whisper()
    inputs = proc(wave, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    ids = mod.generate(inputs.input_features.to(dev), max_new_tokens=64)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text


def get_text_bundle() -> dict | None:
    """Lädt checkpoints/text_emotion.pt + roberta-base für die wörtliche Text-Emotion & Text-Valenz."""
    global text_bundle
    if text_bundle is not None:
        return text_bundle
    path = CHECKPOINT_DIR / "text_emotion.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "states" not in ckpt:
        return None
    model_name = ckpt.get("model_name", "roberta-base")
    dev = pick_device()
    print(f"Lade Text-Encoder {model_name} auf {dev} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(dev).eval()
    mean = np.asarray(ckpt["mean"], dtype="float32")
    std = np.asarray(ckpt["std"], dtype="float32")
    emotions = list(ckpt["emotions"])
    heads = []
    for state in ckpt["states"]:
        m = TextEmotionValenceNet(len(mean), len(emotions))
        m.load_state_dict(state)
        m.eval()
        heads.append(m)
    text_bundle = {
        "tokenizer": tokenizer,
        "encoder": encoder,
        "device": dev,
        "mean": mean,
        "std": std,
        "heads": heads,
        "emotions": emotions,
        "temperature": float(ckpt.get("temperature", 1.0)),
    }
    return text_bundle


@torch.inference_mode()
def predict_text(text: str) -> dict | None:
    """Klassifiziert einen englischen Satz nach wörtlicher Text-Emotion und kontinuierlicher Valenz (-1..+1)."""
    text = text.strip()
    if not text:
        return None
    tb = get_text_bundle()
    if tb is None:
        return None
    enc = tb["tokenizer"]([text], padding=True, truncation=True, max_length=96, return_tensors="pt").to(tb["device"])
    out = tb["encoder"](**enc, output_hidden_states=True)
    cls_vec = out.hidden_states[-1][:, 0, :].float()
    last4 = torch.stack(out.hidden_states[-4:]).mean(dim=0).float()
    mask = enc.attention_mask.unsqueeze(-1).float()
    mean_pool = (last4 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    vec = torch.cat([cls_vec, mean_pool], dim=1).cpu().numpy().astype("float16").astype("float32")
    x = torch.from_numpy((vec - tb["mean"]) / tb["std"]).float()

    logits_list, val_list = [], []
    for head in tb["heads"]:
        l, v = head(x)
        logits_list.append(l.numpy())
        val_list.append(v.numpy())
    mean_logits = np.mean(logits_list, axis=0)
    probs = softmax_np(mean_logits, tb["temperature"])[0]
    valence = float(np.mean(val_list))
    emotions = tb["emotions"]
    prob_dict = {e: float(p) for e, p in zip(emotions, probs)}
    return {
        "text": text,
        "probs": prob_dict,
        "top": max(prob_dict, key=prob_dict.get),
        "valence": round(valence, 3),
    }


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
    if "audeering" in needed and "audeering" not in cache:
        mod, proc, dev = get_audeering()
        a = embed_audeering(mod, proc, dev, wave)
        cache["audeering"] = a[None, :]  # (1, 2051); letzte 3 Werte sind [arousal, dominance, valence]
    if "prosody" in needed and "prosody" not in cache:
        p = extract_prosody(wave).astype("float16")
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        cache["prosody"] = p[None, :]  # (1, 20)


def load_prior_stats(features: list[str], exclude_corpora: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
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

    if "states" in ckpt and "features" in ckpt:
        features = list(ckpt["features"])
        Unsupported = [f for f in features if f not in {"hubert_ms", "wav2vec_ms", "audeering", "prosody"}]
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

    tensors = []
    n_spk = len(all_clips)
    for i, feat_name in enumerate(bundle["features"]):
        x = clip_features[feat_name].astype("float32")
        if bundle["speaker_norm"]:
            stack = np.stack([c["features"][feat_name].astype("float32") for c in all_clips], axis=0)
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


def extract_voice_vad(clip_features: dict[str, np.ndarray], voice_probs: dict[str, float]) -> dict[str, float]:
    """Berechnet Stimm-Valenz (-1..+1), Arousal (0..1) und Dominanz (0..1)."""
    emo_val = voice_probs.get("joy", 0.0) - voice_probs.get("anger", 0.0) - voice_probs.get("sadness", 0.0)
    if "audeering" in clip_features:
        vad = clip_features["audeering"][0, -3:].astype("float32")
        arousal = float(np.clip(vad[0], 0.0, 1.0))
        dominance = float(np.clip(vad[1], 0.0, 1.0))
        aud_val = float(np.clip(vad[2], 0.0, 1.0)) * 2.0 - 1.0
        valence = 0.65 * emo_val + 0.35 * aud_val
    else:
        arousal = dominance = 0.5
        valence = emo_val
    return {
        "valence": round(float(np.clip(valence, -1.0, 1.0)), 3),
        "arousal": round(arousal, 3),
        "dominance": round(dominance, 3),
    }


def analyze_discrepancy(voice_top: str, voice_probs: dict[str, float], voice_vad: dict[str, float], text_res: dict | None) -> dict | None:
    """Vergleicht Stimm-Emotion/Valenz mit wörtlicher Text-Emotion/Valenz (Ironie/Sarkasmus-Indikator)."""
    if text_res is None:
        return None
    v_val = voice_vad["valence"]
    t_val = text_res["valence"]
    gap = round(t_val - v_val, 3)  # positiv = Text positiver als Stimme
    abs_gap = abs(gap)
    t_top = text_res["top"]

    if t_val >= 0.20 and v_val <= -0.20:
        level = "high"
        summary = (
            f"Starke Diskrepanz (Sarkasmus-Muster): Der Wortlaut ist positiv ({t_top}, Valenz {t_val:+.2f}), "
            f"aber der Tonfall klingt negativ ({voice_top}, Stimm-Valenz {v_val:+.2f})."
        )
    elif t_val <= -0.20 and v_val >= 0.25:
        level = "medium"
        summary = (
            f"Gegenläufiges Signal: Der Text ist negativ ({t_top}, Valenz {t_val:+.2f}), "
            f"wird aber fröhlich/positiv ausgesprochen ({voice_top}, Stimm-Valenz {v_val:+.2f})."
        )
    elif voice_top != t_top and abs_gap >= 0.45:
        level = "medium"
        summary = (
            f"Deutlicher Unterschied zwischen Wortlaut ({t_top}, Valenz {t_val:+.2f}) "
            f"und Tonfall ({voice_top}, Stimm-Valenz {v_val:+.2f})."
        )
    else:
        level = "low"
        summary = (
            f"Stimme ({voice_top}, Valenz {v_val:+.2f}) und wörtlicher Text ({t_top}, Valenz {t_val:+.2f}) "
            f"passen zueinander."
        )
    return {
        "level": level,
        "valence_gap": gap,
        "voice_valence": v_val,
        "text_valence": t_val,
        "summary": summary,
    }


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
    top_order = [
        "best_hubert_ms_wav2vec_ms_audeering_prosody_nomeld_spknorm",
        "best_hubert_ms_wav2vec_ms_prosody_nomeld_spknorm",
        "best_hubert_ms_wav2vec_ms_audeering_prosody_nomeld",
        "best_hubert_ms_wav2vec_ms_prosody_nomeld",
    ]
    models.sort(
        key=lambda m: (
            top_order.index(m["id"]) if m["id"] in top_order else (10 if m["id"].startswith("best_") else 20),
            m["id"],
        )
    )
    return models


def evaluate_history(bundle: dict) -> list[dict]:
    history = []
    for clip in speaker_clips:
        probs = predict_for_clip(bundle, clip["features"], speaker_clips)
        top = max(probs, key=probs.get)
        vad = extract_voice_vad(clip["features"], probs)
        text_res = clip.get("text_result")
        disc = analyze_discrepancy(top, probs, vad, text_res)
        history.append({
            "id": clip["id"],
            "name": clip["name"],
            "transcript": clip.get("transcript", ""),
            "seconds": clip["seconds"],
            "probs": probs,
            "top": top,
            "voice_vad": vad,
            "text_result": text_res,
            "discrepancy": disc,
        })
    return history


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.get("/api/models")
def list_models():
    return jsonify(
        models=available_models(),
        speaker_clips=len(speaker_clips),
        has_text_model=(CHECKPOINT_DIR / "text_emotion.pt").exists(),
    )


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


@app.post("/api/text_only")
def classify_text_only():
    """Erlaubt das direkte Testen eines englischen Satzes im Text-Emotions- & Valenz-Modell."""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify(error="Bitte gib einen englischen Satz ein."), 400
    with infer_lock:
        res = predict_text(text)
    if res is None:
        return jsonify(error="Text-Modell (checkpoints/text_emotion.pt) nicht gefunden."), 404
    return jsonify(res)


@app.post("/api/reclassify")
def reclassify():
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
    user_text = request.args.get("name", "").strip()
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
            needed = set(bundle["features"]) | {"hubert_ms", "wav2vec_ms", "audeering", "prosody"}
            feat_cache: dict[str, np.ndarray] = {}
            extract_needed_features(wave, needed, feat_cache)

            # Falls der Nutzer keinen Satz eingetippt hat, per Whisper automatisch transkribieren
            transcript = user_text
            if not transcript:
                try:
                    transcript = transcribe_english(wave)
                except Exception as e:
                    print(f"Whisper-Hinweis: {e}")
                    transcript = ""

            text_res = predict_text(transcript) if transcript else None
            display_name = transcript if transcript else f"Aussage #{next_clip_id}"

            clip_entry = {
                "id": next_clip_id,
                "name": display_name,
                "transcript": transcript,
                "text_result": text_res,
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
    parser.add_argument("--preload", action="store_true", help="Alle Audio- und Text-Encoder direkt beim Start laden")
    args = parser.parse_args()

    if args.preload:
        for key in ENCODERS:
            get_encoder(key)
        get_audeering()
        get_whisper()
        get_text_bundle()
    print(f"\nDemo läuft auf http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True)
