#!/usr/bin/env bash
# ============================================================
# run_server.sh — Lancement du serveur Football Analytics (macOS / Linux)
# Accessible sur http://localhost:8000
# ============================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "============================================================"
echo " Football Analytics Server — Veo Analytics Clone"
echo "============================================================"
echo ""

# Vérifier et activer le virtualenv si présent
if [ -d "venv" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
fi

# Vérifier les modèles
if [ ! -f "models/player_detection.pt" ]; then
    echo "[INFO] Modèles absents. Téléchargement en cours..."
    python3 download_models.py
fi

echo "[INFO] Vérification accélération matérielle..."
python3 -c "import torch; cuda=torch.cuda.is_available(); mps=hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(); print(f'  CUDA: {cuda} | MPS (Apple Silicon): {mps}')" 2>/dev/null || true

echo ""
echo "[INFO] Démarrage du serveur sur http://0.0.0.0:8000"
echo "[INFO] Interface Web : http://localhost:8000"
echo "[INFO] Interface API : http://localhost:8000/docs"
echo "[INFO] Appuyez sur Ctrl+C pour arrêter"
echo ""

uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop asyncio \
    --timeout-keep-alive 300 \
    --log-level info
