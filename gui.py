import sys
import asyncio
import signal

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView  # ！！！required to load Qt6WebEngine DLLs for QtWidgets，太逆天了Qt6
from qasync import QEventLoop

from ui import MonikaGUI


def main():
    
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # 确保并行加载期间有足够的 worker
    import os
    import concurrent.futures
    _cpu_n = os.cpu_count() or 4
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(
        max_workers=max(8, _cpu_n + 4),
        thread_name_prefix="monika-worker",
    ))

    # 深色样式！将来可能还能有别的主题，一切接口都是有的，参考 theme
    from ui.theme import get_stylesheet
    app.setStyleSheet(get_stylesheet())

    window = MonikaGUI()
    window.show()

    # 允许 Ctrl+C 从控制台终止 GUI的思路：
    # Qt 事件循环会屏蔽 Python 的 SIGINT，用一个空定时器每 200ms 唤醒 Python解释器，
    # 让 signal handler 有机会执行，不是特别好的思路
    signal.signal(signal.SIGINT, lambda *_: window.close())
    _sig_timer = QTimer()
    _sig_timer.start(200)
    _sig_timer.timeout.connect(lambda: None)  # 空回调，只为唤醒 Python

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
