import sys
import asyncio
import signal

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView  # ！！！required to load Qt6WebEngine DLLs for QtWidgets，太逆天了Qt6
from qasync import QEventLoop

from ui.theme import get_stylesheet


def main():
    
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # 确保并行加载期间有足够的 worker
    import os
    import concurrent.futures
    _cpu_n = os.cpu_count() or 4
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(
        max_workers=max(16, _cpu_n * 3),
        thread_name_prefix="monika-worker",
    ))

    # 深色样式
    app.setStyleSheet(get_stylesheet())

    # 允许 Ctrl+C 从控制台终止 GUI
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    _sig_timer = QTimer()
    _sig_timer.start(200)
    _sig_timer.timeout.connect(lambda: None)

    # -- 启动加载窗口 ------------------------------------------
    from ui.splash_window import SplashWindow
    from ui import MonikaGUI

    splash = SplashWindow()
    splash.show()

    _main_window = None

    def _on_init_done(components, error):
        nonlocal _main_window
        if error:
            return
        # 创建主窗口，传入预加载的组件
        _main_window = MonikaGUI(components=components)
        # 先关 splash，再首次 show 主窗口（触发系统自带的打开动画）
        splash.accept()
        _main_window.show()
        _main_window.raise_()
        _main_window.activateWindow()
        # Live2D 在窗口 show 之后由 _inject_components 异步加载，加载完成直接渲染

    splash._on_complete = _on_init_done
    splash.start_init()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
