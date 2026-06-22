import os
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import QUrl, QTimer
from PyQt6.QtGui import QColor, QResizeEvent
import logging

from utils.config_loader import get_config, set_config

logger = logging.getLogger(__name__)

class Live2DWidget(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 消除 WebEngine 默认白色底（必须在 load 之前调用）
        self.page().setBackgroundColor(QColor('#1e1e1e'))
        # 设置背景为深灰（VSCode 风格）
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # 允许跨路径加载本地资源 (加载 HTML 和 Model 的跨域限制)
        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        
        self._model_loaded = False        # 是否已成功调用过 loadModel
        self._current_model_path = None   # 当前加载的模型路径
        self._pending_transform = None    # (scale, x, y) 外部注入的变换，优先于 config

        # 加载本控件同级目录 web/live2d_viewer.html
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(ui_dir, "web", "live2d_viewer.html")
        
        if os.path.exists(html_path):
            self.load(QUrl.fromLocalFile(html_path))
            logger.info(f"[Live2D] 已加载 HTML: {html_path}")
        else:
            logger.error(f"[Live2D] 找不到 HTML: {html_path}")
        

    def save_transform(self):
        '''读取 JS 当前变换并写入 config.yaml（在窗口关闭前调用）'''
        def _save(result):
            if not result:
                return
            try:
                set_config("ui.live2d_scale", round(float(result["scale"]), 4))
                set_config("ui.live2d_x", round(float(result["x"]), 2))
                set_config("ui.live2d_y", round(float(result["y"]), 2))
                logger.info("[Live2D] 变换已保存: scale=%.3f x=%.1f y=%.1f",
                            result["scale"], result["x"], result["y"])
            except Exception as e:
                logger.warning("[Live2D] 保存变换失败: %s", e)
        self.page().runJavaScript("window.getModelTransform()", _save)

    def load_model(self, model_path_or_name: str):
        '''传入模型文件夹名称或绝对路径加载模型'''
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # 尝试多种路径匹配方式
        final_path = None
        
        # 方式1：直接给定的是绝对路径且以 .model3.json 结尾
        if os.path.isabs(model_path_or_name) and model_path_or_name.endswith('.model3.json'):
            if os.path.exists(model_path_or_name):
                final_path = model_path_or_name
                logger.info(f"[Live2D] 方式1: 直接绝对路径 {final_path}")
        
        # 方式2：直接给定的是绝对路径但缺少文件名
        elif os.path.isabs(model_path_or_name):
            # 如果是目录，尝试找 .model3.json
            if os.path.isdir(model_path_or_name):
                for file in os.listdir(model_path_or_name):
                    if file.endswith('.model3.json'):
                        final_path = os.path.join(model_path_or_name, file)
                        logger.info(f"[Live2D] 方式2: 目录搜索找到 {final_path}")
                        break
            # 或者当作基础路径，尝试加 .model3.json
            elif not model_path_or_name.endswith('.json'):
                candidate = model_path_or_name + '.model3.json'
                if os.path.exists(candidate):
                    final_path = candidate
                    logger.info(f"[Live2D] 方式3: 补全扩展名 {final_path}")
        
        # 方式4：相对路径或仅名字 (传统方式)
        if not final_path:
            # 尝试 assets/live2d/<name>/<name>.model3.json
            candidate = os.path.join(project_root, "assets", "live2d", model_path_or_name, 
                                    f"{model_path_or_name}.model3.json")
            if os.path.exists(candidate):
                final_path = candidate
                logger.info(f"[Live2D] 方式4: 标准路径 {final_path}")
            else:
                # 尝试在 assets/live2d/<name>/ 下找任何 .model3.json
                folder = os.path.join(project_root, "assets", "live2d", model_path_or_name)
                if os.path.isdir(folder):
                    for root, dirs, files in os.walk(folder):
                        for file in files:
                            if file.endswith('.model3.json'):
                                final_path = os.path.join(root, file)
                                logger.info(f"[Live2D] 方式5: 递归搜索找到 {final_path}")
                                break
                        if final_path:
                            break
        
        if not final_path:
            logger.error(f"[Live2D] 无法找到模型: {model_path_or_name}")
            self.page().runJavaScript(f"debug('ERROR: 找不到模型: {model_path_or_name}');")
            return
        
        # 验证文件存在
        if not os.path.exists(final_path):
            logger.error(f"[Live2D] 模型文件不存在: {final_path}")
            self.page().runJavaScript(f"debug('ERROR: 模型文件不存在');")
            return
        
        # 修正用于 file:/// 协议的路径格式
        model_url = final_path.replace('\\', '/')
        self._current_model_path = model_path_or_name  # 记录路径以便 ensure_rendered 重载
        
        logger.info(f"[Live2D] 加载模型: {model_url}")
        
        # 使用 QTimer 替代 threading.Timer（线程安全）
        def _do_load():
            logger.debug("[Live2D] 执行延迟加载 (500ms 后)")
            # 优先使用外部注入的变换（模式切换时由 main_window 设定）
            if self._pending_transform is not None:
                scale, x, y = self._pending_transform
                self._pending_transform = None
            else:
                scale = get_config("ui.live2d_scale", None)
                x     = get_config("ui.live2d_x",     None)
                y     = get_config("ui.live2d_y",     None)
            if scale is not None and x is not None and y is not None:
                self.page().runJavaScript(
                    f"setInitialTransform({scale}, {x}, {y});"
                )
            self.page().runJavaScript(f"loadModel('file:///{model_url}');")
            self._model_loaded = True
            logger.debug("[Live2D] loadModel 已调用")
        
        # 延迟 4000ms（4秒）以确保 CDN 脚本已加载
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(_do_load)
        timer.start(500)  # 500ms
        
        # 保存 timer 引用以防止被垃圾回收
        self._load_timer = timer
        logger.info(f"[Live2D] 计时器已启动，将在 500ms 后调用 loadModel")

    def ensure_rendered(self):
        '''显示 Live2D 面板后调用，确保模型已加载并触发重绘
        若尚未加载，则从 config 中读取路径并加载；若已加载，则触发一次 resize 刷新'''
        model_path = get_config("ui.live2d_model", "")
        if not self._model_loaded or (model_path and model_path != self._current_model_path):
            if model_path:
                logger.info("[Live2D] ensure_rendered: 触发模型加载")
                self.load_model(model_path)
            return
        # 已加载：触发 resize 让 WebGL 重新布局（解决 show() 后 canvas 尺寸为 0 的问题）
        QTimer.singleShot(100, lambda: self.page().runJavaScript(
            "window.onWindowResize && window.onWindowResize();"
        ))

    def set_emotion(self, emotion: str):
        '''切换表情None/'normal'/'idle' 等中性词会重置为模型默认表情'''
        logger.debug(f"[Live2D] 设置表情: {emotion}")
        if not emotion or emotion.lower() in ("normal", "idle", "neutral", "none", "null"):
            self.page().runJavaScript("window.setEmotion && window.setEmotion(null);")
        else:
            escaped = emotion.replace("\\", "\\\\").replace("'", "\\'")
            self.page().runJavaScript(f"window.setEmotion && window.setEmotion('{escaped}');")

    def reset_expression(self):
        '''重置表情到模型默认状态（等同于 set_emotion(None)）'''
        self.page().runJavaScript("window.setEmotion && window.setEmotion(null);")

    def trigger_motion(self, name: str, group: str = None):
        '''触发动作（motion3）name 为空时静默跳过'''
        if not name:
            return
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        group_arg = f"'{group}'" if group else "null"
        self.page().runJavaScript(
            f"window.triggerMotion && window.triggerMotion('{escaped}', {group_arg});"
        )

    def set_talking(self, is_talking: bool):
        '''开启/关闭说话口型动画'''
        state = 'true' if is_talking else 'false'
        logger.debug(f"[Live2D] 设置说话状态: {is_talking}")
        self.page().runJavaScript(f"window.setTalking && window.setTalking({state});")

    def set_mouth_params(self, params: dict):
        '''直接设置多个嘴型 Param* 参数（由口型同步逐帧驱动）
        params: dict[str, float]，key = Live2D Param ID（如 ParamMouthOpenY）
        '''
        import json
        self.page().runJavaScript(
            f"window.setMouthParams && window.setMouthParams({json.dumps(params)});"
        )

    def apply_transform(self, scale: float, x: float, y: float):
        '''运行时直接应用变换到已加载的模型（模式切换/全屏切换时调用）
        与 setInitialTransform 不同，此方法会立即修改 currentModel 的位置和缩放
        '''
        self.page().runJavaScript(
            f"window.applyTransform && window.applyTransform({scale}, {x}, {y});"
        )
