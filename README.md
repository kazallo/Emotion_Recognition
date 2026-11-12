# Emotion_Irony_Recognition
Recognizing emotions and sarcasm in spoken English from both tone of voice and words.

## Idee
Mehrere kleine Modelle, die jeweils einen Aspekt erkennen, und ein Ironie-Klassifikator darüber.
Ironie entsteht oft aus dem Widerspruch zwischen Kanälen (z. B. positiver Text, genervter Ton).

| Modell | Status |
|---|---|
| Emotion aus Stimme (SVM, MLP auf wav2vec/HuBERT) | fertig |
| Emotion aus Text | geplant |
| Sentiment/Polarität aus Text | geplant |
| Prosodie (Pitch, Tempo, Betonung) | geplant |
| Kontext (vorherige Äußerung) | geplant |
| Fusion / Ironie-Klassifikator (z. B. auf MUStARD) | geplant |

## Ablauf
```
uv run data_preprocessing/build_combined_csv.py       # data/combined.csv (Splits pro Sprecher)
uv run features/extract_wav2vec.py combined           # oder extract_DistilHUBERT.py
uv run models/train_mlp.py combined --embedding hubert [--mode mix]
uv run models/train_svm.py combined --wav2vec
uv run models/evaluate.py ravdess mlp_hubert
```
Training und Evaluation schreiben Kennzahlen nach `results/<name>.json`
(UAR gesamt und pro Korpus). Daten (`data/`) und Modelle (`checkpoints/`) liegen nicht im Repository.

## Datensätze
CREMA-D, RAVDESS, ESD, MELD (Klassen: anger, joy, neutral, sadness). Metrik: UAR (Unweighted Average Recall).
