#!/usr/bin/env bash
# Kopiert das Projekt (ohne die 35 GB unnötigen MELD-Rohdaten) auf den zugewiesenen DSAI-GPU-Server
# und nutzt dabei automatisch den lokalen SSH-Schlüssel ./key.
set -euo pipefail

# Deine HSD-Kennung und der vom Compute-Dispatcher (https://compute.dsai.medien.intern.hs-duesseldorf.de/)
# zugewiesene Host (z. B. 10.50.60.151) bzw. SSH-Port:
SERVER_USER="theo.zech@study.hs-duesseldorf.de"
SERVER_HOST="${SERVER_HOST:-HIER_HOST_IP_AUS_APPROVAL_MAIL_EINTRAGEN}"
SERVER_PORT="${SERVER_PORT:-22}"
SSH_KEY="./key"
REMOTE_DIR="~/Emotion_Irony_Recognition"

chmod 600 "$SSH_KEY"
SSH_CMD="ssh -i $SSH_KEY -p $SERVER_PORT"

MODE="${1:-push}"

if [ "$SERVER_HOST" = "HIER_HOST_IP_AUS_APPROVAL_MAIL_EINTRAGEN" ]; then
    echo "Bitte trage in sync_and_run_server.sh (Zeile 9) bei SERVER_HOST die IP des GPU-Servers ein,"
    echo "die du vom DSAI Compute-Dispatcher (https://compute.dsai.medien.intern.hs-duesseldorf.de/) bekommst (z. B. 10.50.60.151)!"
    exit 1
fi

if [ "$MODE" = "pull" ]; then
    echo "=== Hole fertige Checkpoints und Ergebnisse vom Server ($SERVER_USER@$SERVER_HOST) ==="
    rsync -avz --progress -e "$SSH_CMD" "$SERVER_USER@$SERVER_HOST:$REMOTE_DIR/checkpoints/" ./checkpoints/
    rsync -avz --progress -e "$SSH_CMD" "$SERVER_USER@$SERVER_HOST:$REMOTE_DIR/results/" ./results/
    exit 0
fi

echo "=== 1/2 Kopiere Code und benötigte Daten auf $SERVER_USER@$SERVER_HOST ==="
rsync -avz --progress -e "$SSH_CMD" \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'key' \
    --exclude 'key.pub' \
    --exclude 'data/MELD.Raw*' \
    --exclude 'data/meld_audio' \
    --exclude 'data/Audio_Speech_Actors_01-24' \
    --exclude 'data/Emotion Speech Dataset/000*' \
    --exclude 'data/Emotion Speech Dataset/0010' \
    --exclude 'data/combined_hubert.npy' \
    --exclude 'data/combined_wav2vec.npy' \
    --exclude 'data/cremad_*.npy' \
    --exclude 'data/ravdess_*.npy' \
    ./ "$SERVER_USER@$SERVER_HOST:$REMOTE_DIR/"

echo ""
echo "=== 2/2 Übertragung fertig! ==="
echo "Logge dich jetzt im Terminal auf dem Server ein und starte die Berechnung:"
echo "  $SSH_CMD $SERVER_USER@$SERVER_HOST"
echo "  cd $REMOTE_DIR && nohup bash run_server.sh > server_run.log 2>&1 &"
echo "  tail -f server_run.log"
