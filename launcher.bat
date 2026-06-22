@echo off
setlocal enabledelayedexpansion

REM 禁用快速编辑模式，防止点击窗口时程序暂停
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f >nul 2>&1

cls
echo.
echo ============================================================
echo     Monika-V3 Launcher
echo ============================================================
echo.

echo [*] Required services (check if running):
echo   - Ollama (http://localhost:11434)
echo   - TTS (http://127.0.0.1:5000)
echo   - VTube Studio
echo   - Voicemeeter
echo.

echo [*] Starting Monika...
echo.

REM 创建临时 Python 脚本，用于最小化 launcher 窗口
(
echo import ctypes, time
echo time.sleep(2^)
echo hWnd = ctypes.windll.kernel32.GetConsoleWindow(^)
echo ctypes.windll.user32.ShowWindow(hWnd, 6^)
) > "%temp%\minimize_launcher.py"

REM 在后台运行最小化脚本
start "" /b python "%temp%\minimize_launcher.py" >nul 2>&1

REM 启动 GUI（前台阻塞）
python gui.py

REM 清理临时文件
del "%temp%\minimize_launcher.py" >nul 2>&1

REM GUI 关闭后自动关闭 launcher
exit /b 0
