@echo off
setlocal enabledelayedexpansion
chcp 65001

cls
echo.
echo ============================================================
echo     Monika-V3 Setup Script
echo ============================================================
echo.

REM -- Check Python ---------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11 or 3.12.
    echo Download: https://www.python.org/downloads/release/python-3119/
    echo          During install, check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PY_VER=%%V
echo [OK] Python %PY_VER% found
for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do (
    if not ("%%A.%%B"=="3.11" or "%%A.%%B"=="3.12") (
        echo [ERROR] Python 3.11 or 3.12 is required. Found: %PY_VER%
        echo         The bundled editdistance wheel only supports cp311.
        echo         Please install Python 3.11 or 3.12: https://www.python.org/downloads/release/python-3119/
        pause
        exit /b 1
    )
)

REM -- Check Git ------------------------------------------------
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Please install Git for Windows.
    echo Download: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)
echo [OK] Git found
echo.

REM -- Step 1: SenseVoiceSmall (STT model) ----------------------
echo ============================================================
echo  Step 1: Download SenseVoiceSmall (speech recognition, ~1GB)
echo ============================================================
echo.

if exist modules\SenseVoiceSmall (
    echo [SKIP] modules\SenseVoiceSmall already exists
) else (
    echo [*] Trying ModelScope (recommended for China)...
    git clone https://www.modelscope.cn/iic/SenseVoiceSmall.git modules\SenseVoiceSmall 2>nul
    if errorlevel 1 (
        echo [WARN] ModelScope failed, trying HuggingFace mirror (hf-mirror.com)...
        git clone https://hf-mirror.com/FunAudioLLM/SenseVoiceSmall modules\SenseVoiceSmall 2>nul
        if errorlevel 1 (
            echo [WARN] Mirror failed, trying official HuggingFace...
            git clone https://huggingface.co/FunAudioLLM/SenseVoiceSmall modules\SenseVoiceSmall
            if errorlevel 1 (
                echo [ERROR] Could not clone SenseVoiceSmall.
                echo         Please manually download and place in modules\SenseVoiceSmall
                pause
                exit /b 1
            )
        )
    )
    echo [OK] SenseVoiceSmall downloaded
)
echo.

REM -- Step 2: GPT-SoVITS v2pro ---------------------------------
echo ============================================================
echo  Step 2: GPT-SoVITS v2pro (TTS engine)
echo ============================================================
echo.
echo  IMPORTANT: GPT-SoVITS ships as a packaged release with its own
echo  Python runtime. A plain git clone is NOT sufficient.
echo.
if exist modules\GPT-SoVITS-v2pro-20250604\runtime\python.exe (
    echo [SKIP] modules\GPT-SoVITS-v2pro-20250604 already present with runtime
) else (
    echo [ACTION REQUIRED]
    echo   1. Open: https://github.com/RVC-Boss/GPT-SoVITS/releases
    echo   2. Download the archive: GPT-SoVITS-v2pro-20250604-*.7z
    echo   3. Extract so the path exists:
    echo        modules\GPT-SoVITS-v2pro-20250604\runtime\python.exe
    echo.
    echo   After extracting, press any key to continue setup.
    pause
    if not exist modules\GPT-SoVITS-v2pro-20250604\runtime\python.exe (
        echo [ERROR] runtime\python.exe still not found. Exiting.
        pause
        exit /b 1
    )
)

echo [*] Copying custom TTS server scripts...
copy /Y modules\tts_server\tts_v3_server.py modules\GPT-SoVITS-v2pro-20250604\tts_v3_server.py >nul 2>&1
copy /Y modules\tts_server\tts_v3_server-v2pro.py modules\GPT-SoVITS-v2pro-20250604\tts_v3_server-v2pro.py >nul 2>&1
echo [OK] TTS server scripts copied
echo.

REM -- Step 3: Install dependencies -----------------------------
echo ============================================================
echo  Step 3: Install Python dependencies
echo ============================================================
echo.

echo [*] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel --quiet
echo.

REM -- Detect CUDA version and install llama-cpp-python ---------
echo [*] Detecting GPU for llama-cpp-python wheel selection...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO] No NVIDIA GPU detected. Installing CPU-only llama-cpp-python.
    echo        LLM inference will run on CPU (slower).
    pip install llama-cpp-python --quiet
) else (
    for /f "tokens=9" %%C in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set CUDA_RAW=%%C
    if not defined CUDA_RAW set CUDA_RAW=12.4
    for /f "tokens=1,2 delims=." %%A in ("!CUDA_RAW!") do set CUDA_TAG=cu%%A%%B
    echo [INFO] CUDA !CUDA_RAW! detected, trying wheel for !CUDA_TAG!...
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/!CUDA_TAG! --quiet 2>nul
    if errorlevel 1 (
        echo [WARN] Wheel for !CUDA_TAG! not found, trying cu124 (works with CUDA 12.x/13.x)...
        pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --quiet 2>nul
        if errorlevel 1 (
            echo [WARN] Pre-built wheel unavailable. Falling back to source build.
            echo        This requires Visual C++ Build Tools:
            echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
            pip install llama-cpp-python
        )
    )
)
if errorlevel 1 (
    echo [ERROR] llama-cpp-python installation failed.
    pause
    exit /b 1
)
echo [OK] llama-cpp-python installed
echo.

REM -- editdistance: bundled wheel (C++ source has Windows MSVC issues) --
echo [*] Installing editdistance from bundled wheel...
pip install editdistance --no-index --find-links=wheels --quiet
if errorlevel 1 (
    echo [ERROR] editdistance wheel install failed.
    echo 请检查 wheels\ 目录下是否有匹配当前 Python 和系统的 whl 文件
    pause
    exit /b 1
)
echo [OK] editdistance installed
echo.

REM -- Remaining requirements ------------------------------------
echo [*] Installing remaining dependencies...
echo     (torch, funasr, chromadb, PyQt6 - may take 10-30 minutes)
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo.
    echo Common solutions:
    echo   - PyTorch CUDA mismatch: install PyTorch manually first
    echo     https://pytorch.org/get-started/locally/
    echo     then re-run setup.bat
    echo   - Network issue: check connection and retry
    echo.
    pause
    exit /b 1
)
echo.
echo ============================================================
echo  Setup completed successfully!
echo ============================================================
echo.
echo  Next steps:
echo.
echo  1. Place your LLM model:
echo       GGUF:   assets\model\your-model.gguf
echo               set brain.gguf.model_path in config.yaml
echo       Ollama: ollama pull your-model-name
echo               set brain.ollama.model_name in config.yaml
echo.
echo  2. Place GPT-SoVITS model weights:
echo       modules\GPT-SoVITS-v2pro-20250604\GPT_weights_v2Pro\
echo       modules\GPT-SoVITS-v2pro-20250604\SoVITS_weights_v2Pro\
echo.
echo  3. Place reference audio (10s, clean WAV):
echo       assets\wav\ref\your-audio.wav
echo       Update tts.ref_audio_path and tts.ref_text in config.yaml
echo.
echo  4. Launch:
echo       python gui.py      <- recommended (GUI + built-in Live2D)
echo       python main.py     <- terminal mode
echo.
echo  5. VTube Studio and Voicemeeter are OPTIONAL.
echo       The GUI has a built-in Live2D renderer.
echo       Voicemeeter is only needed for mic/speaker routing.
echo.
echo.
pause