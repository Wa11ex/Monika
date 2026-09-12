#!/bin/bash
# Monika-V3 Launcher for Linux

echo "============================================================"
echo "    Monika-V3 Launcher (Linux)"
echo "============================================================"
echo ""

# Check virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[WARN] Virtual environment not found. Please run ./setup.sh first."
fi

echo "[*] Required services (check if running):"
echo "  - Ollama (http://localhost:11434)"
echo "  - TTS (http://127.0.0.1:5000)"
echo ""

# Start TTS server assuming GPT-SoVITS environment is configured globally or same venv
# Note: On Linux, GPT-SoVITS is usually set up with conda or an external python.
# Here we just remind the user, or try to run it if the script exists
if [ -f "modules/GPT-SoVITS-v2pro-20250604/tts_v3_server.py" ]; then
    echo "[*] Remember to start TTS server in a separate terminal: python modules/GPT-SoVITS-v2pro-20250604/tts_v3_server.py"
fi

echo "[*] Starting Monika..."
echo ""

# Run GUI
python gui.py
