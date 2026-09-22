#!/usr/bin/env bash
# Script di installazione automatica per Water Ring Alpha Extractor.
# Non richiede conoscenze di programmazione: esegui semplicemente
#   ./install.sh
# e lo script fara' tutto da solo (crea un ambiente Python isolato,
# installa le librerie necessarie e verifica/installa FFmpeg).
set -e

echo "=============================================="
echo " Water Ring Alpha Extractor - Installazione"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Verifica Python 3
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERRORE: Python 3 non trovato sul sistema."
    echo "Installa Python 3 da https://www.python.org/downloads/ e riesegui questo script."
    exit 1
fi
echo "-> Python 3 trovato: $(python3 --version)"

# 2. Verifica/installa FFmpeg (necessario per leggere/scrivere i video)
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "-> FFmpeg non trovato. Provo a installarlo automaticamente..."
    if command -v apt-get >/dev/null 2>&1; then
        echo "   (verra' chiesta la password di amministratore per 'sudo apt-get install ffmpeg')"
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    else
        echo "ERRORE: impossibile installare FFmpeg automaticamente su questo sistema."
        echo "Installa FFmpeg manualmente da https://ffmpeg.org/download.html e riesegui questo script."
        exit 1
    fi
else
    echo "-> FFmpeg gia' presente: $(ffmpeg -version | head -n1)"
fi

# 3. Crea un ambiente Python isolato (non tocca il resto del sistema)
if [ ! -d ".venv" ]; then
    echo "-> Creo l'ambiente Python isolato (.venv)..."
    python3 -m venv .venv
fi

echo "-> Installo le librerie necessarie (puo' richiedere qualche minuto)..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=============================================="
echo " Installazione completata!"
echo " Per avviare il programma esegui: ./run.sh"
echo "=============================================="
