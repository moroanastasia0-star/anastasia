#!/usr/bin/env bash
# Avvia l'interfaccia locale del Water Ring Alpha Extractor.
# Esegui semplicemente:  ./run.sh
# Si aprira' automaticamente una pagina nel browser.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Ambiente non trovato. Eseguo prima l'installazione..."
    ./install.sh
fi

source .venv/bin/activate
python3 ui/app.py
