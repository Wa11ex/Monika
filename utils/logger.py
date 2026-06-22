'''
统一日志模块
'''

import glob
import logging
import os
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_initialized = False
_MAX_SESSIONS = 3  # 保留最近 N 次会话日志


def _cleanup_old_logs(log_dir: str) -> None:
    '''删除旧日志文件，仅保留最近 _MAX_SESSIONS 个'''
    pattern = os.path.join(log_dir, "monika-*.log")
    files = sorted(glob.glob(pattern))
    for old in files[:-(_MAX_SESSIONS - 1)] if len(files) >= _MAX_SESSIONS else []:
        try:
            os.remove(old)
        except OSError:
            pass


def setup_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    '''
    初始化全局日志，懒加载
    '''
    global _initialized
    if _initialized:
        return
    _initialized = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s %(levelname)-5s %(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # -- 终端打印 --
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # -- 写文件：按会话归档，保留最近 3 次 --
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(_PROJECT_ROOT, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    _cleanup_old_logs(log_dir)

    session_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"monika-{session_ts}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # -- 降低第三方库噪音 --
    for noisy in ("httpx", "httpcore", "aiohttp.access", "urllib3", "modelscope",
                   "qasync._windows", "qasync._QEventLoop",
                   
                   # ddgs / web fetch 底层库
                   "duckduckgo_search", "ddgs", "primp", "rustls", "h2",
                   "hyper_util", "reqwest", "cookie_store",
                   "hickory_net", "hickory_resolver", "hickory_proto",
                   
                   # HTTP/2 帧编解码（hpack）
                   "hpack",
                   
                   # ChromaDB / sentence-transformers 内部噪音
                   "chromadb.telemetry", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    '''获取命名 logger，供各模块顶层使用'''
    return logging.getLogger(name)


class QtLogHandler(logging.Handler):
    '''
    将日志记录转发到 Qt 文本组件的 Handler，给 GUI 使用
    在 GUI 初始化完成后注册到 root logger
    '''

    def __init__(self, callback, min_level: int = logging.INFO):
        super().__init__()
        self._callback = callback
        self.setLevel(min_level)
        self.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._callback(msg)
        except Exception:
            self.handleError(record)
