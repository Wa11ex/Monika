#!/bin/bash
# Monika-V3 Setup Script for Linux

echo "============================================================"
echo "    Monika-V3 Setup Script (Linux)"
echo "============================================================"
echo ""

# -- 1. Check Python ----------------------------------------------
if command -v python3.11 &>/dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.12 &>/dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    echo "[ERROR] Python 3 not found. Please install Python 3.11 or 3.12."
    exit 1
fi

PY_VER=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PY_VER" != "3.11" && "$PY_VER" != "3.12" ]]; then
    echo "[ERROR] Python 3.11 or 3.12 is required. Found: $PY_VER"
    exit 1
fi
echo "[OK] Python $PY_VER found"

# -- 2. Create venv -----------------------------------------------
echo "[*] Creating virtual environment (venv)..."
$PYTHON_CMD -m venv venv
source venv/bin/activate
echo "[OK] Virtual environment activated"

# -- 3. Init config -----------------------------------------------
if [ ! -f "config.yaml" ]; then
    echo "[*] Creating config.yaml from example..."
    cp "config example.yaml" config.yaml
fi

# -- 4. SenseVoiceSmall (STT model) -------------------------------
echo ""
echo "============================================================"
echo " Step 1: Download SenseVoiceSmall (speech recognition)"
echo "============================================================"
if [ -d "modules/SenseVoiceSmall" ]; then
    echo "[SKIP] modules/SenseVoiceSmall already exists"
else
    echo "[*] Trying ModelScope..."
    git clone https://www.modelscope.cn/iic/SenseVoiceSmall.git modules/SenseVoiceSmall || \
    git clone https://hf-mirror.com/FunAudioLLM/SenseVoiceSmall modules/SenseVoiceSmall || \
    git clone https://huggingface.co/FunAudioLLM/SenseVoiceSmall modules/SenseVoiceSmall
fi

# -- 5. GPT-SoVITS -----------------------------------------------
echo ""
echo "============================================================"
echo " Step 2: GPT-SoVITS (TTS engine)"
echo "============================================================"
echo " IMPORTANT: For Linux, you should manually setup GPT-SoVITS-v2pro."
echo " Or run it in a separate Docker container."
echo " Copying custom TTS server scripts assuming the directory exists..."
mkdir -p modules/GPT-SoVITS-v2pro-20250604
cp modules/tts_server/tts_v3_server.py modules/GPT-SoVITS-v2pro-20250604/tts_v3_server.py 2>/dev/null
cp modules/tts_server/tts_v3_server-v2pro.py modules/GPT-SoVITS-v2pro-20250604/tts_v3_server-v2pro.py 2>/dev/null

# -- 6. Install Python Dependencies -------------------------------
echo ""
echo "============================================================"
echo " Step 3: Install Python dependencies"
echo "============================================================"
python -m pip install --upgrade pip setuptools wheel

echo "[*] Detecting GPU for llama-cpp-python..."
if command -v nvidia-smi &>/dev/null; then
    CUDA_RAW=$(nvidia-smi | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | awk '{print $3}')
    echo "[INFO] CUDA $CUDA_RAW detected."
    # On Linux, building from source is often easier with pip and CUDA toolkit installed,
    # or using the wheels provided by abetlen.
    CUDA_TAG="cu$(echo $CUDA_RAW | tr -d '.')"
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/$CUDA_TAG || \
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 || \
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
else
    echo "[INFO] No NVIDIA GPU detected. Installing CPU-only llama-cpp-python."
    pip install llama-cpp-python
fi

echo "[*] Installing editdistance (compiling from source on Linux)..."
pip install editdistance

echo "[*] Installing remaining dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo " Setup completed successfully!"
echo "============================================================"
echo " Run ./launcher.sh to start Monika"
