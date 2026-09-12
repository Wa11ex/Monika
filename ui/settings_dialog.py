'''
设置对话框（全量 config 覆盖）
'''

import os
import yaml
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout, QLabel,
    QDialogButtonBox, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QKeySequenceEdit, QTextEdit, QHBoxLayout, QPushButton, QFileDialog
)
from PyQt6.QtGui import QKeySequence
from PyQt6 import QtCore

from utils.config_loader import load_config, get_config
from utils.model_registry import list_models
from .utils import scrolled

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置面板内联小按钮的覆盖样式（覆盖全局 11pt / padding:8px 16px）
_SMALL_BTN_STYLE = "font-size: 9pt; font-weight: normal; padding: 2px 8px;"

class SafeDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置点击后才获取焦点
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        # 只有在拥有焦点的情况下才允许滚动
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            # 否则将事件传播给父容器（让页面可以正常滚动）
            event.ignore()
            
class SafeSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置点击后才获取焦点
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        # 只有在拥有焦点的情况下才允许滚动
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            # 否则将事件传播给父容器（让页面可以正常滚动）
            event.ignore()

class SafeComboBox(QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置点击后才获取焦点
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        # 只有在拥有焦点的情况下才允许滚动
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            # 否则将事件传播给父容器（让页面可以正常滚动）
            event.ignore()


class SettingsDialog(QDialog):
    '''设置对话框，编辑并保存所有 config.yaml 参数'''

    def __init__(self, parent=None, vts_manager=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(560, 640)

        # VTS 管理器引用（None 表示尚未初始化）
        self._vts_manager = vts_manager
        # 用户在对话框内请求的 VTS 操作 (None / "connect" / "disconnect")
        self.vts_requested_action = None

        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            self._raw = yaml.safe_load(f) or {}

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(scrolled(self._build_audio_tab()),  "麦克风")
        tabs.addTab(scrolled(self._build_brain_tab()),  "大脑")
        tabs.addTab(self._build_persona_tab(),          "人设")
        tabs.addTab(scrolled(self._build_tts_tab()),    "TTS")
        tabs.addTab(scrolled(self._build_system_tab()), "系统")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_debug_mode_changed(self, state: int) -> None:
        '''调试日志复选框：即时更新内存配置并重绘聊天框，无需保存'''
        from utils.config_loader import set_config
        val = self._debug_mode.isChecked()
        set_config("ui.debug_mode", val)
        main = self.parent()
        if hasattr(main, "_redraw_chat"):
            main._redraw_chat()

    # -- 工具 ----------------------------------------------
    @staticmethod
    def _mk_form():
        '''创建标准表单'''
        w = QWidget()
        f = QFormLayout(w)
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        return w, f

    @staticmethod
    def _populate_cameras(combo, current: int):
        '''异步枚举摄像头并填充下拉框（避免打开设置时主线程卡顿）'''
        from PyQt6.QtCore import QTimer

        # 先填当前设备占位，避免下拉框空白
        combo.addItem(f"摄像头 {current}", current)

        def _work():
            try:
                from core.vision.camera import list_cameras
                cams = list_cameras()
            except Exception:
                cams = []
            if not cams:
                cams = [(current, f"摄像头 {current}")]

            def _apply():
                try:
                    combo.clear()
                    for idx, name in cams:
                        combo.addItem(name, idx)
                    for i in range(combo.count()):
                        if combo.itemData(i) == current:
                            combo.setCurrentIndex(i)
                            break
                except Exception:
                    pass  # 对话框已关闭时忽略
            QTimer.singleShot(0, _apply)

        import threading
        threading.Thread(target=_work, daemon=True).start()

    def _refresh_model_combo(self):
        '''从注册表刷新 GGUF 模型下拉列表，并选中当前 config 中的模型'''
        saved_path = self._g("brain", "gguf", "model_path", default="")
        self._model_combo.clear()
        models = list_models()
        default_idx = 0
        for i, m in enumerate(models):
            label = f"{m['name']}  [{m['size_gb']:.1f} GB]"
            self._model_combo.addItem(label, userData=m["path"])
            if m["path"] == saved_path or m["name"] == os.path.splitext(
                os.path.basename(saved_path)
            )[0]:
                default_idx = i
        if not models:
            self._model_combo.addItem("（无可用模型，请先导入）", userData="")
        self._model_combo.setCurrentIndex(default_idx)

    def _on_import_model(self):
        '''打开模型导入对话框，导入成功后刷新下拉列表'''
        from .model_import_dialog import ModelImportDialog
        dlg = ModelImportDialog(parent=self)
        if dlg.exec() == ModelImportDialog.DialogCode.Accepted:
            self._refresh_model_combo()
            # 自动选中刚导入的模型
            if dlg.imported_model_name:
                for i in range(self._model_combo.count()):
                    if dlg.imported_model_name in self._model_combo.itemText(i):
                        self._model_combo.setCurrentIndex(i)
                        break

    def _populate_mic_devices(self):
        '''枚举系统音频输入设备并填充下拉列表'''
        saved_id = self._g("ears", "mic_id")  # int or None
        self._mic_device_combo.clear()
        self._mic_device_combo.addItem("系统默认", userData=None)
        default_idx = 0  # 下拉框中的默认选中项
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    name = f"{i}: {dev['name']}"
                    self._mic_device_combo.addItem(name, userData=i)
                    if saved_id is not None and i == int(saved_id):
                        default_idx = self._mic_device_combo.count() - 1
        except Exception:
            pass  # sounddevice 不可用时，只保留"系统默认"
        self._mic_device_combo.setCurrentIndex(default_idx)

    def _populate_output_devices(self, combo: SafeComboBox, saved_id):
        '''枚举系统音频输出设备并填充下拉列表'''
        combo.clear()
        combo.addItem("系统默认", userData=None)
        default_idx = 0
        try:
            import sounddevice as sd
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_output_channels", 0) > 0:
                    name = f"{i}: {dev['name']}"
                    combo.addItem(name, userData=i)
                    if saved_id is not None and i == int(saved_id):
                        default_idx = combo.count() - 1
        except Exception:
            pass  # sounddevice 不可用时，只保留"系统默认"
        combo.setCurrentIndex(default_idx)

    def _g(self, *keys, default=None):
        '''安全获取嵌套字典值'''
        d = self._raw
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, None)
            else:
                return default
            if d is None:
                return default
        return d

    @staticmethod
    def _sep(label: str) -> QLabel:
        '''分隔符标签'''
        lbl = QLabel(f"-- {label} --")
        lbl.setStyleSheet("color: gray; font-size: 11px; margin-top: 6px;")
        return lbl

    def _with_browse(self, line_edit: QLineEdit, is_dir=False, title="选择文件", filter="*.*") -> QWidget:
        '''为输入框添加浏览按钮'''
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(4)
        l.addWidget(line_edit)
        
        btn = QPushButton("浏览...")
        btn.setFixedWidth(60)
        btn.setStyleSheet(_SMALL_BTN_STYLE)
        
        def _browse():
            start_dir = line_edit.text() or PROJECT_ROOT
            if is_dir:
                path = QFileDialog.getExistingDirectory(self, title, start_dir)
            else:
                path, _ = QFileDialog.getOpenFileName(self, title, start_dir, filter)
            if path:
                line_edit.setText(path)
                
        btn.clicked.connect(_browse)
        l.addWidget(btn)
        return w

    # -- Tab 1: 麦克风 & 语音 ------------------------------
    def _build_audio_tab(self):
        w, form = self._mk_form()

        form.addRow(self._sep("输入模式"))
        self._input_mode = SafeComboBox()
        self._input_mode.addItems(["typing", "mic_always_on", "mic_push_to_talk"])
        self._input_mode.setCurrentText(self._g("audio", "input_mode", default="typing"))
        form.addRow("输入模式:", self._input_mode)

        self._ptt_key = QKeySequenceEdit()
        self._ptt_key.setKeySequence(QKeySequence(self._g("audio", "ptt_key", default="Space")))
        form.addRow("PTT 按键:", self._ptt_key)

        form.addRow(self._sep("麦克风设备"))
        self._mic_device_combo = SafeComboBox()
        self._mic_device_combo.setMinimumWidth(260)
        self._populate_mic_devices()
        form.addRow("麦克风设备:", self._mic_device_combo)

        self._ears_device = SafeComboBox()
        self._ears_device.addItems(["cpu", "cuda"])
        self._ears_device.setCurrentText(self._g("ears", "device", default="cpu"))
        form.addRow("ASR 运算设备:", self._ears_device)

        self._ears_model = QLineEdit(self._g("ears", "model_name", default="modules/SenseVoiceSmall"))
        form.addRow("ASR 模型路径:", self._with_browse(self._ears_model, is_dir=True, title="选择 ASR 模型目录"))

        form.addRow(self._sep("VAD 参数"))
        self._vad = SafeDoubleSpinBox()
        self._vad.setRange(0.001, 0.5); self._vad.setSingleStep(0.001); self._vad.setDecimals(4)
        self._vad.setValue(self._g("ears", "vad_threshold", default=0.02))
        form.addRow("VAD 阈值:", self._vad)

        self._ivad = SafeDoubleSpinBox()
        self._ivad.setRange(0.001, 0.5); self._ivad.setSingleStep(0.001); self._ivad.setDecimals(4)
        self._ivad.setValue(self._g("ears", "interrupt_vad_threshold", default=0.015))
        form.addRow("打断 VAD 阈值:", self._ivad)

        self._silence = SafeDoubleSpinBox()
        self._silence.setRange(0.1, 5.0); self._silence.setSingleStep(0.1); self._silence.setDecimals(2)
        self._silence.setValue(self._g("ears", "silence_limit", default=0.5))
        form.addRow("静音判定 (秒):", self._silence)

        self._pre_buf = SafeSpinBox()
        self._pre_buf.setRange(1, 20)
        self._pre_buf.setValue(self._g("ears", "pre_buffer_len", default=4))
        form.addRow("预录制块数:", self._pre_buf)

        form.addRow(self._sep("降噪 & 回声消除"))
        self._deepfilter = QCheckBox()
        self._deepfilter.setChecked(self._g("ears", "echo_cancellation", "enable_deepfilter", default=True))
        form.addRow("启用 DeepFilterNet:", self._deepfilter)

        self._loopback_combo = SafeComboBox()
        self._loopback_combo.setMinimumWidth(260)
        self._populate_output_devices(
            self._loopback_combo,
            self._g("ears", "echo_cancellation", "loopback_device_id")
        )
        form.addRow("EC 参考输出设备:", self._loopback_combo)

        form.addRow(self._sep("情感检测"))
        self._emotion_en = QCheckBox()
        self._emotion_en.setChecked(self._g("ears", "emotion", "enabled", default=True))
        form.addRow("启用情感检测:", self._emotion_en)

        self._emotion_ctx = QCheckBox()
        self._emotion_ctx.setChecked(self._g("ears", "emotion", "add_to_context", default=True))
        form.addRow("情感传递给 LLM:", self._emotion_ctx)

        return w

    # -- Tab 2: 大脑 ---------------------------------------
    def _build_brain_tab(self):
        w, form = self._mk_form()

        self._backend = SafeComboBox()
        self._backend.addItems(["gguf", "ollama"])
        self._backend.setCurrentText(self._g("brain", "backend", default="gguf"))
        form.addRow("LLM 后端:", self._backend)

        form.addRow(self._sep("GGUF 参数"))

        # -- 模型选择下拉 + 导入按钮 --------------------------
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self._model_combo = SafeComboBox()
        self._model_combo.setMinimumWidth(220)
        self._refresh_model_combo()
        model_row.addWidget(self._model_combo, 1)

        import_btn = QPushButton("导入模型…")
        import_btn.setStyleSheet(_SMALL_BTN_STYLE)
        import_btn.clicked.connect(self._on_import_model)
        model_row.addWidget(import_btn)

        model_row_widget = QWidget()
        model_row_widget.setLayout(model_row)
        form.addRow("已导入模型:", model_row_widget)

        self._gpu_layers = SafeSpinBox()
        self._gpu_layers.setRange(-1, 200)
        self._gpu_layers.setValue(self._g("brain", "gguf", "n_gpu_layers", default=-1))
        form.addRow("GPU 层数 (-1=全量):", self._gpu_layers)

        self._n_ctx = SafeSpinBox()
        self._n_ctx.setRange(512, 65536); self._n_ctx.setSingleStep(512)
        self._n_ctx.setValue(self._g("brain", "gguf", "n_ctx", default=2048))
        form.addRow("上下文长度 (n_ctx):", self._n_ctx)

        self._n_batch = SafeSpinBox()
        self._n_batch.setRange(64, 4096); self._n_batch.setSingleStep(64)
        self._n_batch.setValue(self._g("brain", "gguf", "n_batch", default=512))
        form.addRow("批处理大小 (n_batch):", self._n_batch)

        self._gguf_temp = SafeDoubleSpinBox()
        self._gguf_temp.setRange(0.0, 2.0); self._gguf_temp.setSingleStep(0.05); self._gguf_temp.setDecimals(2)
        self._gguf_temp.setValue(self._g("brain", "gguf", "temperature", default=0.95))
        form.addRow("Temperature:", self._gguf_temp)

        self._top_p = SafeDoubleSpinBox()
        self._top_p.setRange(0.0, 1.0); self._top_p.setSingleStep(0.05); self._top_p.setDecimals(2)
        self._top_p.setValue(self._g("brain", "gguf", "top_p", default=0.85))
        form.addRow("Top-P:", self._top_p)
        
        self._top_k = SafeDoubleSpinBox()
        self._top_k.setRange(0, 100); self._top_k.setSingleStep(1); self._top_k.setDecimals(0)
        self._top_k.setValue(self._g("brain", "gguf", "top_k", default=10))
        form.addRow("Top-K:", self._top_k)

        self._min_p = SafeDoubleSpinBox()
        self._min_p.setRange(0.0, 1.0); self._min_p.setSingleStep(0.05); self._min_p.setDecimals(2)
        self._min_p.setValue(self._g("brain", "gguf", "min_p", default=0.0))
        form.addRow("Min-P:", self._min_p)
        
        self._repeat = SafeDoubleSpinBox()
        self._repeat.setRange(1.0, 2.0); self._repeat.setSingleStep(0.05); self._repeat.setDecimals(2)
        self._repeat.setValue(self._g("brain", "gguf", "repeat_penalty", default=1.1))
        form.addRow("Repeat Penalty:", self._repeat)

        self._gguf_turns = SafeSpinBox()
        self._gguf_turns.setRange(2, 100)
        self._gguf_turns.setValue(self._g("brain", "gguf", "max_history_turns", default=14))
        form.addRow("最大历史轮数:", self._gguf_turns)

        self._enable_thinking = QCheckBox()
        self._enable_thinking.setChecked(bool(self._g("brain", "gguf", "enable_thinking", default=False)))
        self._enable_thinking.setToolTip(
            "启用 Qwen3/Qwen3.5 的 Thinking 模式（思维链）\n"
            "开启后模型会在回答前进行内部推理，有助于复杂问题，但会增加延迟\n"
            "<think> 内容不会朗读，仅在开启调试日志时显示于控制台"
        )
        form.addRow("启用思维链 (Thinking):", self._enable_thinking)

        self._think_budget = SafeSpinBox()
        self._think_budget.setRange(0, 10000)
        self._think_budget.setSingleStep(64)
        self._think_budget.setValue(self._g("brain", "gguf", "think_budget", default=0))
        self._think_budget.setToolTip("思维链最大字符数，0=不限制超出预算后思考内容被截断")
        form.addRow("思维链预算 (字符):", self._think_budget)

        self._external_template = QLineEdit(self._g("brain", "gguf", "external_template", default=""))
        form.addRow("Jinja 模板路径:",
                    self._with_browse(self._external_template, title="选择 Jinja 模板",
                                      filter="Jinja Template (*.jinja);;All (*.*)"))

        form.addRow(self._sep("Ollama 参数"))  # noqa: E501
        self._ollama_url = QLineEdit(self._g("brain", "ollama", "base_url", default=""))
        form.addRow("base_url:", self._ollama_url)

        self._ollama_model = QLineEdit(self._g("brain", "ollama", "model_name", default=""))
        form.addRow("模型名:", self._ollama_model)

        self._ollama_temp = SafeDoubleSpinBox()
        self._ollama_temp.setRange(0.0, 2.0); self._ollama_temp.setSingleStep(0.05); self._ollama_temp.setDecimals(2)
        self._ollama_temp.setValue(self._g("brain", "ollama", "temperature", default=0.85))
        form.addRow("Temperature:", self._ollama_temp)

        self._ollama_turns = SafeSpinBox()
        self._ollama_turns.setRange(2, 100)
        self._ollama_turns.setValue(self._g("brain", "ollama", "max_history_turns", default=14))
        form.addRow("最大历史轮数:", self._ollama_turns)

        self._ollama_think_budget = SafeSpinBox()
        self._ollama_think_budget.setRange(0, 10000)
        self._ollama_think_budget.setSingleStep(64)
        self._ollama_think_budget.setValue(self._g("brain", "ollama", "think_budget", default=0))
        self._ollama_think_budget.setToolTip("思维链最大字符数，0=不限制超出预算后思考内容被截断")
        form.addRow("思维链预算 (字符):", self._ollama_think_budget)

        return w

    # -- Tab 3: 人设 ---------------------------------------
    def _build_persona_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel("角色系统提示 (character.prompt):"))
        self._char_prompt = QTextEdit()
        self._char_prompt.setPlaceholderText("在这里输入角色系统提示...")
        self._char_prompt.setPlainText(self._g("brain", "character", "prompt") or "")
        layout.addWidget(self._char_prompt, 3)

        layout.addWidget(QLabel("示例对话 (character.examples):"))
        self._char_examples = QTextEdit()
        self._char_examples.setPlaceholderText("在这里输入示例对话...")
        self._char_examples.setPlainText(self._g("brain", "character", "examples") or "")
        layout.addWidget(self._char_examples, 2)

        return w

    # -- Tab 4: TTS ----------------------------------------
    def _build_tts_tab(self):
        w, form = self._mk_form()

        self._tts_url = QLineEdit(self._g("tts", "server_url", default=""))
        form.addRow("TTS 流式 URL:", self._tts_url)

        self._tts_health = QLineEdit(self._g("tts", "health_url", default=""))
        form.addRow("健康检查 URL:", self._tts_health)

        self._tts_sr = SafeSpinBox()
        self._tts_sr.setRange(8000, 48000); self._tts_sr.setSingleStep(100)
        self._tts_sr.setValue(self._g("tts", "sample_rate", default=24000))
        form.addRow("采样率:", self._tts_sr)

        self._ref_audio = QLineEdit(self._g("tts", "ref_audio_path", default=""))
        form.addRow("参考音频路径:", self._with_browse(self._ref_audio, title="选择参考音频", filter="Audio (*.wav *.mp3);;All (*.*)"))

        self._ref_text = QLineEdit(self._g("tts", "ref_text", default=""))
        form.addRow("参考文本:", self._ref_text)

        self._text_lang = SafeComboBox()
        self._text_lang.addItems(["zh", "en", "ja", "ko", "auto"])
        self._text_lang.setCurrentText(self._g("tts", "text_lang", default="zh"))
        form.addRow("输出语言:", self._text_lang)

        self._ref_lang = SafeComboBox()
        self._ref_lang.addItems(["zh", "en", "ja", "ko", "auto"])
        self._ref_lang.setCurrentText(self._g("tts", "ref_lang", default="zh"))
        form.addRow("参考音频语言:", self._ref_lang)

        self._if_sr = QCheckBox()
        self._if_sr.setChecked(bool(self._g("tts", "if_sr", default=True)))
        form.addRow("启用超分辨率 (if_sr):", self._if_sr)

        self._speed = SafeDoubleSpinBox()
        self._speed.setRange(0.3, 3.0); self._speed.setSingleStep(0.05); self._speed.setDecimals(2)
        self._speed.setValue(float(self._g("tts", "speed", default=1.0)))
        form.addRow("语速 (speed):", self._speed)

        return w

    # -- Tab 5: 系统 ---------------------------------------
    def _build_system_tab(self):
        w, form = self._mk_form()

        form.addRow(self._sep("UI 及渲染"))
        self._live2d_model = QLineEdit(self._g("ui", "live2d_model", default="Mk3_v2.2__by_0x4682B4_"))
        form.addRow("Live2D 模型 (名字或JSON):", self._with_browse(self._live2d_model, title="选择 Live2D 配置文件", filter="Live2D Model (*.model3.json);;All (*.*)"))

        self._debug_mode = QCheckBox()
        self._debug_mode.setChecked(self._g("ui", "debug_mode", default=True))
        form.addRow("显示调试日志:", self._debug_mode)
        # 即时热插拔：勾选状态变化时立即生效，无需点击「保存」
        self._debug_mode.stateChanged.connect(self._on_debug_mode_changed)

        form.addRow(self._sep("音频路由"))
        self._audio_dev_combo = SafeComboBox()
        self._audio_dev_combo.setMinimumWidth(260)
        self._populate_output_devices(self._audio_dev_combo, self._g("audio", "device_id"))
        form.addRow("音频输出设备:", self._audio_dev_combo)

        form.addRow(self._sep("VTube Studio"))
        self._vts_plugin = QLineEdit(self._g("vts", "plugin_name", default="Monika_Core"))
        form.addRow("插件名:", self._vts_plugin)
        self._vts_token = QLineEdit(self._g("vts", "token_path", default="./vts_token.txt"))
        form.addRow("Token 路径:", self._with_browse(self._vts_token, title="选择 Token 文件", filter="Text (*.txt);;All (*.*)"))

        # VTS 连接状态 + 操作按钮
        vts_ctrl_row = QHBoxLayout()
        vts_ctrl_row.setContentsMargins(0, 0, 0, 0)
        self._vts_status_label = QLabel()
        self._vts_connect_btn    = QPushButton()
        self._vts_disconnect_btn  = QPushButton("断开 VTS")
        self._vts_disconnect_btn.setFixedWidth(90)
        self._vts_connect_btn.setFixedWidth(90)
        self._vts_connect_btn.setStyleSheet(_SMALL_BTN_STYLE)
        self._vts_disconnect_btn.setStyleSheet(_SMALL_BTN_STYLE)
        self._refresh_vts_ui()
        self._vts_connect_btn.clicked.connect(self._on_vts_connect)
        self._vts_disconnect_btn.clicked.connect(self._on_vts_disconnect)
        vts_ctrl_row.addWidget(self._vts_status_label)
        vts_ctrl_row.addStretch()
        vts_ctrl_row.addWidget(self._vts_connect_btn)
        vts_ctrl_row.addWidget(self._vts_disconnect_btn)
        vts_ctrl_widget = QWidget()
        vts_ctrl_widget.setLayout(vts_ctrl_row)
        form.addRow("连接状态:", vts_ctrl_widget)

        form.addRow(self._sep("上下文 & 时间"))
        self._ctx_time_en = QCheckBox()
        self._ctx_time_en.setChecked(self._g("context", "time", "enabled", default=True))
        form.addRow("注入当前时间:", self._ctx_time_en)
        self._ctx_time_fmt = QLineEdit(self._g("context", "time", "format", default="%Y/%m/%d %A %H:%M"))
        form.addRow("时间格式:", self._ctx_time_fmt)

        form.addRow(self._sep("长期记忆"))
        self._mem_en = QCheckBox()
        self._mem_en.setChecked(self._g("memory", "enabled", default=False))
        form.addRow("启用对话记忆:", self._mem_en)
        self._lore_en = QCheckBox()
        self._lore_en.setChecked(self._g("memory", "lore", "enabled", default=False))
        self._lore_en.setToolTip("独立于对话记忆，仅控制前世记忆（Lore）的语义检索\n关闭对话记忆后仍可单独启用")
        form.addRow("启用前世记忆 (Lore):", self._lore_en)
        self._mem_turns = SafeSpinBox()
        self._mem_turns.setRange(2, 100)
        self._mem_turns.setValue(self._g("memory", "session", "max_turns", default=14))
        form.addRow("Session 最大轮数:", self._mem_turns)

        form.addRow(self._sep("声纹识别"))
        self._spk_en = QCheckBox()
        self._spk_en.setChecked(self._g("speaker_id", "enabled", default=False))
        form.addRow("启用声纹识别:", self._spk_en)
        self._spk_threshold = SafeDoubleSpinBox()
        self._spk_threshold.setRange(0.0, 1.0); self._spk_threshold.setSingleStep(0.01); self._spk_threshold.setDecimals(2)
        self._spk_threshold.setValue(self._g("speaker_id", "threshold", default=0.65))
        form.addRow("声纹置信度阈值:", self._spk_threshold)

        form.addRow(self._sep("主动发言"))
        self._po_en = QCheckBox()
        self._po_en.setChecked(self._g("proactive_output", "enabled", default=True))
        form.addRow("启用主动发言:", self._po_en)

        form.addRow(self._sep("多模态"))
        self._vision_en = QCheckBox()
        self._vision_en.setChecked(self._g("multimodal", "vision", "enabled", default=False))
        form.addRow("启用视觉 (摄像头):", self._vision_en)

        self._vision_dev_combo = SafeComboBox()
        self._populate_cameras(
            self._vision_dev_combo,
            self._g("multimodal", "vision", "capture_device", default=0),
        )
        form.addRow("摄像头设备:", self._vision_dev_combo)

        self._vision_fps = SafeSpinBox()
        self._vision_fps.setRange(1, 60)
        self._vision_fps.setValue(self._g("multimodal", "vision", "preview_fps", default=15))
        form.addRow("摄像头预览 fps:", self._vision_fps)

        self._screen_en = QCheckBox()
        self._screen_en.setChecked(self._g("multimodal", "screen", "enabled", default=False))
        form.addRow("启用屏幕 OCR:", self._screen_en)

        self._screen_fps = SafeSpinBox()
        self._screen_fps.setRange(1, 30)
        self._screen_fps.setValue(self._g("multimodal", "screen", "preview_fps", default=5))
        form.addRow("屏幕预览 fps:", self._screen_fps)

        return w

    # -- VTS 连接控制 ----------------------------------------
    def _refresh_vts_ui(self):
        '''根据 vts_manager.is_connected 更新状态标签和按钮状态'''
        connected = bool(self._vts_manager and self._vts_manager.is_connected)
        if self._vts_manager is None:
            self._vts_status_label.setText("● 未初始化")
            self._vts_status_label.setStyleSheet("color: #888;")
        elif connected:
            self._vts_status_label.setText("● 已连接")
            self._vts_status_label.setStyleSheet("color: #55dd88; font-weight: bold;")
        else:
            self._vts_status_label.setText("○ 未连接")
            self._vts_status_label.setStyleSheet("color: #aaa;")
        self._vts_connect_btn.setText("同步 VTS" if not connected else "重新同步")
        self._vts_connect_btn.setEnabled(self._vts_manager is not None)
        self._vts_disconnect_btn.setEnabled(connected)

    def _on_vts_connect(self):
        '''用户点击"同步VTS" -> 保存配置后关闭对话框，由主窗口执行异步连接'''
        self.vts_requested_action = "connect"
        self.accept()   # 触发保存流程

    def _on_vts_disconnect(self):
        '''用户点击"断开VTS" -> 保存配置后关闭对话框，由主窗口执行异步断开'''
        self.vts_requested_action = "disconnect"
        self.accept()

    # -- 保存 ----------------------------------------------
    def _on_save(self):
        raw = self._raw

        # audio
        a = raw.setdefault("audio", {})
        a["input_mode"]     = self._input_mode.currentText()
        a["ptt_key"]        = self._ptt_key.keySequence().toString()
        a["device_id"] = self._audio_dev_combo.currentData()

        # ears
        e = raw.setdefault("ears", {})
        e["mic_id"]                  = self._mic_device_combo.currentData()  # int or None
        e["device"]                  = self._ears_device.currentText()
        e["model_name"]              = self._ears_model.text().strip()
        e["vad_threshold"]           = round(self._vad.value(), 4)
        e["interrupt_vad_threshold"] = round(self._ivad.value(), 4)
        e["silence_limit"]           = round(self._silence.value(), 2)
        e["pre_buffer_len"]          = self._pre_buf.value()
        ec = e.setdefault("echo_cancellation", {})
        ec["enable_deepfilter"] = self._deepfilter.isChecked()
        ec["loopback_device_id"] = self._loopback_combo.currentData()  # int or None
        ec["loopback_keyword"]   = ""  # 已改用 device_id，清空旧关键字
        em = e.setdefault("emotion", {})
        em["enabled"]        = self._emotion_en.isChecked()
        em["add_to_context"] = self._emotion_ctx.isChecked()

        # brain
        b = raw.setdefault("brain", {})
        b["backend"] = self._backend.currentText()
        g = b.setdefault("gguf", {})
        g["model_path"]        = self._model_combo.currentData() or ""
        g["n_gpu_layers"]      = self._gpu_layers.value()
        g["n_ctx"]             = self._n_ctx.value()
        g["n_batch"]           = self._n_batch.value()
        g["temperature"]       = round(self._gguf_temp.value(), 2)
        g["top_p"]             = round(self._top_p.value(), 2)
        g["top_k"]             = self._top_k.value()
        g["min_p"]             = round(self._min_p.value(), 2)
        g["repeat_penalty"]    = round(self._repeat.value(), 2)
        g["max_history_turns"] = self._gguf_turns.value()
        g["enable_thinking"]   = self._enable_thinking.isChecked()
        g["think_budget"]      = self._think_budget.value()
        g["external_template"] = self._external_template.text().strip()
        o = b.setdefault("ollama", {})
        o["base_url"]          = self._ollama_url.text().strip()
        o["model_name"]        = self._ollama_model.text().strip()
        o["temperature"]       = round(self._ollama_temp.value(), 2)
        o["max_history_turns"] = self._ollama_turns.value()
        o["think_budget"]      = self._ollama_think_budget.value()
        ch = b.setdefault("character", {})
        ch["prompt"]   = self._char_prompt.toPlainText()
        ch["examples"] = self._char_examples.toPlainText()

        # ui
        u = raw.setdefault("ui", {})
        u["live2d_model"] = self._live2d_model.text().strip()
        u["debug_mode"]   = self._debug_mode.isChecked()

        # tts
        t = raw.setdefault("tts", {})
        t["server_url"]     = self._tts_url.text().strip()
        t["health_url"]     = self._tts_health.text().strip()
        t["sample_rate"]    = self._tts_sr.value()
        t["ref_audio_path"] = self._ref_audio.text().strip()
        t["ref_text"]       = self._ref_text.text().strip()
        t["text_lang"]      = self._text_lang.currentText()
        t["ref_lang"]       = self._ref_lang.currentText()
        t["if_sr"]          = self._if_sr.isChecked()
        t["speed"]          = round(self._speed.value(), 2)

        # # ui
        # u = raw.setdefault("ui", {})
        # u["live2d_model"] = self._live2d_model.text().strip()

        # vts
        v = raw.setdefault("vts", {})
        v["plugin_name"] = self._vts_plugin.text().strip()
        v["token_path"]  = self._vts_token.text().strip()

        # context
        raw.setdefault("context", {}).setdefault("time", {}).update({
            "enabled": self._ctx_time_en.isChecked(),
            "format":  self._ctx_time_fmt.text().strip(),
        })

        # memory
        m = raw.setdefault("memory", {})
        m["enabled"] = self._mem_en.isChecked()
        m.setdefault("lore", {})["enabled"] = self._lore_en.isChecked()
        m.setdefault("session", {})["max_turns"] = self._mem_turns.value()

        # speaker_id
        s = raw.setdefault("speaker_id", {})
        s["enabled"]   = self._spk_en.isChecked()
        s["threshold"] = round(self._spk_threshold.value(), 2)

        # proactive_output
        raw.setdefault("proactive_output", {})["enabled"] = self._po_en.isChecked()

        # multimodal
        mm = raw.setdefault("multimodal", {})
        mm.setdefault("vision", {})["enabled"] = self._vision_en.isChecked()
        mm["vision"]["capture_device"] = self._vision_dev_combo.currentData()
        mm["vision"]["preview_fps"] = self._vision_fps.value()
        mm.setdefault("screen", {})["enabled"] = self._screen_en.isChecked()
        mm["screen"]["preview_fps"] = self._screen_fps.value()

        # 写回文件
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        with open(os.path.join(assets_dir, "user_settings.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        load_config(config_path)
        self.accept()
