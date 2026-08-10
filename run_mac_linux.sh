#!/usr/bin/env bash
# Run this from Terminal with: ./run_mac_linux.sh
# (first time only: chmod +x run_mac_linux.sh)
# Sets up a virtual environment (first run only), installs dependencies,
# and starts the app in your browser.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing/updating dependencies (first run may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "Starting the app... Streamlit will open it in your browser automatically."
echo "Press Ctrl+C in this terminal to stop the app."
echo ""
streamlit run app.py
