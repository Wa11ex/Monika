'''
Monika 统一启动器
自动拉起Ollama、TTS Server、VTube Studio、Voicemeeter
'''

import os
import sys
import subprocess
import asyncio
import time
import atexit
import aiohttp
from utils.config_loader import get_config
from utils.helpers import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

IS_WIN = sys.platform == "win32"


def _popen_flags(detached: bool = False) -> dict:
    '''返回当前平台对应的 subprocess.Popen 启动标志（仅 Windows 需要）'''
    if not IS_WIN:
        return {}
    flags = subprocess.CREATE_NO_WINDOW
    if detached:
        flags |= subprocess.DETACHED_PROCESS
    return {"creationflags": flags}


# 记录所有由 launcher 拉起的子进程，用于优雅（暴力）关闭
_managed_processes: list[subprocess.Popen] = []
_service_ports: list[tuple[int, str]] = []  # (port, label)
_shutdown_done = False

# 全局日志回调钩子
_log_callback = None

def set_log_callback(callback):
    '''设置全局日志回调钩子，签名为 callback(message: str)'''
    global _log_callback
    _log_callback = callback

def _do_log(msg):
    '''写日志 + GUI 回调'''
    log.info("%s", msg)
    if _log_callback:
        _log_callback(msg)




def _is_process_running(name: str) -> bool:
    '''检查指定名称的进程是否已在运行，兼容Linux'''
    if IS_WIN:
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            return name.lower() in result.stdout.lower()
        except Exception:
            return False
    else:
        # Linux: strip .exe suffix, pgrep
        proc_name = name.removesuffix(".exe")
        try:
            result = subprocess.run(["pgrep", "-x", proc_name], capture_output=True)
            return result.returncode == 0
        except Exception:
            return False


async def _wait_for_health(
    url: str,
    timeout: float,
    label: str,
    json_check: dict | None = None
) -> bool:
    '''
    轮询健康检查接口
    json_check: 拓展检查的字段字典
    '''
    start = time.monotonic()
    last_reason = ""
    while time.monotonic() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        if json_check:
                            body = await resp.json()
                            mismatch = [
                                f"{k}={body.get(k)!r}"
                                for k, v in json_check.items()
                                if body.get(k) != v
                            ]
                            if mismatch:
                                last_reason = ", ".join(mismatch)
                            else:
                                _do_log(f"  >>> [{label}] 服务已就绪")
                                return True
                        else:
                            _do_log(f"  >>> [{label}] 服务已就绪")
                            return True
        except Exception:
            pass
        await asyncio.sleep(1)
    reason = f" (最后状态: {last_reason})" if last_reason else ""
    _do_log(f"  !!! [{label}] 等待超时 ({timeout}s){reason}")
    return False


def _launch_app(exe_path: str, label: str) -> subprocess.Popen | None:
    '''启动一个exe'''
    resolved = resolve_path(exe_path)
    exe_name = os.path.basename(resolved)

    if _is_process_running(exe_name):
        _do_log(f"  >>> [{label}] {exe_name} 已在运行，跳过")
        return None

    if not os.path.isfile(resolved):
        _do_log(f"  !!! [{label}] 找不到: {resolved}")
        return None

    _do_log(f"  >>> [{label}] 正在启动 {exe_name}...")
    proc = subprocess.Popen(
        [resolved],
        cwd=os.path.dirname(resolved),
        **_popen_flags(detached=True)
    )
    _managed_processes.append(proc)
    return proc


def _launch_ollama() -> subprocess.Popen | None:
    '''启动 Ollama'''
    cfg = get_config("launcher.ollama", {})
    if not cfg.get("auto_start", False):
        return None

    exe = cfg.get("exe_path", "ollama")

    ollama_proc_name = "ollama.exe" if IS_WIN else "ollama"
    if _is_process_running(ollama_proc_name):
        _do_log("  >>> [Ollama] ollama 已在运行，跳过")
        return None

    _do_log("  >>> [Ollama] 正在启动 ollama serve...")
    try:
        proc = subprocess.Popen(
            [exe, "serve"],
            **_popen_flags()
        )
        _managed_processes.append(proc)
        _service_ports.append((11434, "Ollama"))
        return proc
    except FileNotFoundError:
        _do_log("  !!! [Ollama] 找不到 ollama，请确保已安装并在 PATH 中")
        return None


def _launch_tts_server() -> subprocess.Popen | None:
    '''启动 GPT-SoVITS TTS 服务器（使用其自带的 runtime/python.exe）'''
    cfg = get_config("launcher.tts", {})
    if not cfg.get("auto_start", False):
        return None

    python_exe = resolve_path(cfg.get("python_exe", ""))
    script = resolve_path(cfg.get("script", ""))
    work_dir = resolve_path(cfg.get("work_dir", ""))

    if not os.path.isfile(python_exe):
        _do_log(f"  !!! [TTS] 找不到 Python: {python_exe}")
        return None
    if not os.path.isfile(script):
        _do_log(f"  !!! [TTS] 找不到脚本: {script}")
        return None

    # 清理 5000 端口
    if _is_port_in_use(5000):
        _do_log("  >>> [TTS] 检测到 5000 端口有残留进程，清理中...")
        _kill_by_port(5000, "TTS-残留")
        time.sleep(1.5)

    _do_log(f"  >>> [TTS] 正在启动 TTS Server...")
    _do_log(f"       Python: {os.path.basename(python_exe)}")
    _do_log(f"       Script: {os.path.basename(script)}")
    _do_log(f"       CWD:    {work_dir}")

    proc = subprocess.Popen(
        [python_exe, script],
        cwd=work_dir,
        **_popen_flags()
    )
    _managed_processes.append(proc)
    _service_ports.append((5000, "TTS"))
    return proc


async def launch_all() -> "asyncio.Task | None":
    '''
    launch all
    返回的是 TTS 健康检查
    '''
    _do_log("\n" + "=" * 30)
    _do_log("     Monika Launcher - 开始")
    _do_log("=" * 30)

    # -- 1. 外部 exe --
    apps_cfg = get_config("launcher.apps", {})
    for app_name, app_cfg in apps_cfg.items():
        if app_cfg.get("auto_start", False):
            exe_path = app_cfg.get("exe_path", "")
            if exe_path:
                _launch_app(exe_path, app_name)

    # -- 2. Ollama 启动（await 健康） --
    brain_backend = get_config("brain.backend", "ollama")
    if brain_backend == "gguf":
        log.debug("[Launcher] brain.backend=gguf，跳过 Ollama 检查与启动")
    else:
        ollama_cfg = get_config("launcher.ollama", {})
        if ollama_cfg.get("auto_start", False):
            _launch_ollama()
            health_url = ollama_cfg.get("health_url", "http://localhost:11434/v1/models")
            timeout = ollama_cfg.get("timeout", 30)
            ok = await _wait_for_health(health_url, timeout, "Ollama")
            if not ok:
                _do_log("  !!! [OLLAMA] 启动失败，相关功能可能不可用")

    # -- 3. TTS启动进程，创建健康检查 Task 但【不在此处等待】 --
    #    由 init_monika 与 Brain/Ears 并行 gather，这能节省 ~30s！
    tts_task: asyncio.Task | None = None
    tts_cfg = get_config("launcher.tts", {})
    if tts_cfg.get("auto_start", False):
        _launch_tts_server()
        health_url = tts_cfg.get("health_url", "http://127.0.0.1:5000/health")
        timeout = tts_cfg.get("timeout", 120)
        _do_log(f"  >>> [TTS] 正在启动（将与 Brain/Ears 并行加载，最长 {timeout}s）...")
        tts_task = asyncio.create_task(
            _wait_for_health(health_url, timeout, "TTS", json_check={"model_loaded": True})
        )

    _do_log("=" * 30)
    _do_log(" " * 9 + "Launcher 完成")
    _do_log("=" * 30 + "\n")
    return tts_task


def _kill_tree(pid: int) -> bool:
    '''强制杀死进程树，兼容Linux'''
    if IS_WIN:
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            return result.returncode == 0
        except Exception:
            return False
    else:
        import signal
        try:
            import psutil
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            return True
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
                return True
            except Exception:
                return False


def _is_port_in_use(port: int) -> bool:
    '''检查端口占用，兼容Linux'''
    if IS_WIN:
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            for line in result.stdout.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    return True
            return False
        except Exception:
            return False
    else:
        try:
            # ss 
            result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True)
            return f":{port} " in result.stdout or f":{port}\n" in result.stdout
        except Exception:
            return False


def _kill_by_port(port: int, label: str):
    '''杀占用端口的进程，也兼容Linux'''
    if IS_WIN:
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            for line in result.stdout.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    pid = int(line.strip().split()[-1])
                    if pid > 0:
                        log.warning("[Launcher] [%s] 端口 %d 仍被占用 (PID=%d)，强制终止...", label, port, pid)
                        _kill_tree(pid)
                    return
        except Exception:
            pass
    else:
        # fuser -k 5000/tcp
        try:
            log.warning("[Launcher] [%s] 端口 %d 仍被占用，执行 fuser -k...", label, port)
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
        except Exception:
            pass


def _verify_shutdown(timeout: float = 8.0):
    '''
    等待所有端口真正释放，超时后再次强杀
    打印每个端口的最终状态，可以用来再杀
    '''
    if not _service_ports:
        return

    log.info("[Launcher] 确认服务端口已释放...")
    pending = list(_service_ports)  # [(port, label), ...]
    deadline = time.monotonic() + timeout

    while pending and time.monotonic() < deadline:
        time.sleep(0.5)
        pending = [(p, lbl) for p, lbl in pending if _is_port_in_use(p)]

    # 仍然占用的端口：再次强杀
    for port, label in pending:
        log.warning("[Launcher] [%s] 端口 %d 未释放，执行第二次强杀...", label, port)
        _kill_by_port(port, label)
        time.sleep(1)
        if _is_port_in_use(port):
            log.warning("[Launcher] [%s] 端口 %d 仍然占用，可能需要手动处理", label, port)
        else:
            log.info("[Launcher] [%s] 端口 %d 已释放", label, port)

    # 已释放的端口打印确认
    freed = [lbl for p, lbl in _service_ports if p not in [pp for pp, _ in pending]]
    for lbl in freed:
        log.info("[Launcher] [%s] 端口已确认释放", lbl)


def launch_vts_app() -> bool:
    '''启动 VTube Studio '''
    vts_cfg = get_config("launcher.apps.vts", {})
    exe_path = vts_cfg.get("exe_path", "")
    if not exe_path:
        log.warning("[Launcher] 未配置 launcher.apps.vts.exe_path，无法启动 VTube Studio")
        return False
    result = _launch_app(exe_path, "VTube Studio")
    # result is None if already running (which is still OK), Popen otherwise
    return True


def shutdown_all():
    '''杀进程树 + 端口兜底 + 释放确认'''
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True

    if not _managed_processes and not _service_ports:
        return

    log.info("[Launcher] 正在关闭所有服务...")

    # 1. PID 杀进程树
    for proc in _managed_processes:
        try:
            if proc.poll() is None:
                log.info("[Launcher] 终止进程树 PID=%d", proc.pid)
                _kill_tree(proc.pid)
        except Exception:
            pass

    # 等待 5s最多
    deadline = time.monotonic() + 5
    for proc in _managed_processes:
        remaining = max(0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass

    # 2.确认端口已释放（含第二轮强杀）
    _verify_shutdown(timeout=8.0)

    _managed_processes.clear()
    _service_ports.clear()
    log.info("[Launcher] 关闭流程完成")
    log.info("[Launcher] 所有服务已关闭")


# atexit 保底，包括 Ctrl+C、IDE 关闭、Python 退出等，都会触发
atexit.register(shutdown_all)
