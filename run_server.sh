#!/usr/bin/env bash
# Komplett-Pipeline für den Hochschul-GPU-Server (nutzt automatisch CUDA, falls vorhanden).
# Aufruf direkt:   bash run_server.sh
# Aufruf im Hintergrund (überlebt SSH-Abbruch):  nohup bash run_server.sh > server_run.log 2>&1 &
set -euo pipefail

echo "=== 1/6 MSP-Podcast Embeddings + Valenz/Arousal/Dominanz (audeering) ==="
uv run features/extract_audeering.py combined

echo "=== 2/6 Messehallen-Augmentation (KI-Con: Raumhall + Hintergrundgeräusch) ==="
uv run features/extract_audeering.py combined --augment-con

echo "=== 3/6 Stimm-Emotionsmodell OHNE Sprecher-Norm (Einzelsatz am KI-Con-Stand) ==="
uv run models/train_best.py --features hubert_ms wav2vec_ms audeering prosody --con-augment --seeds 5

echo "=== 4/6 Stimm-Emotionsmodell MIT Sprecher-Norm (mit Kalibrierungssatz am Stand) ==="
uv run models/train_best.py --features hubert_ms wav2vec_ms audeering prosody --speaker-norm --con-augment --seeds 5

echo "=== 5/6 Reiner Text-Datensatz (4 Korpora: CARER, GoEmotions, DailyDialog, EmoContext) ==="
uv run data_preprocessing/build_text_csv.py
uv run features/extract_text.py text_combined

echo "=== 6/6 Text-Emotions- & Valenz-Modell (roberta-base, 5 Seeds + Kalibrierung) ==="
uv run models/train_text.py --seeds 5

echo "=== Fertig! Alle Modelle in checkpoints/ und Kennzahlen in results/ ==="
