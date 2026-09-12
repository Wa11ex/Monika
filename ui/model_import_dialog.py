'''
模型导入对话框

支持两种格式：
  - GGUF 文件：直接复制到 assets/model/ 并注册
  - HF safetensors 文件夹：调用 tools/convert_hf_to_gguf.py 转换为 F16 GGUF 后注册

用法：
    dlg = ModelImportDialog(parent=self)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        # dlg.imported_model_name 含新注册的模型名
        refresh_model_combo()
'''

import os
import asyncio

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar,
    QTextEdit, QFileDialog, QWidget,
)
from PyQt6.QtCore import Qt
from qasync import asyncSlot

# 内联小按钮覆盖样式（与 settings_dialog 保持一致）
_SMALL_BTN = "font-size: 9pt; font-weight: normal; padding: 2px 8px;"


class ModelImportDialog(QDialog):
    '''模型导入对话框
    接受成功后 self.imported_model_name 保存已导入的模型名称
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入模型")
        self.setMinimumWidth(580)
        self.setMinimumHeight(440)
        self.imported_model_name: str | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # -- 源路径 ------------------------------------------
        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        self._src_edit = QLineEdit()
        self._src_edit.setPlaceholderText("选择 GGUF 文件 或 HF safetensors 文件夹...")
        self._src_edit.textChanged.connect(self._on_src_changed)

        btn_file = QPushButton("GGUF…")
        btn_file.setFixedWidth(64)
        btn_file.setStyleSheet(_SMALL_BTN)
        btn_file.setToolTip("选择 .gguf 文件")
        btn_file.clicked.connect(self._browse_file)

        btn_dir = QPushButton("文件夹…")
        btn_dir.setFixedWidth(72)
        btn_dir.setStyleSheet(_SMALL_BTN)
        btn_dir.setToolTip("选择包含 config.json + *.safetensors 的 HF 文件夹")
        btn_dir.clicked.connect(self._browse_dir)

        src_row.addWidget(QLabel("源路径:"))
        src_row.addWidget(self._src_edit, 1)
        src_row.addWidget(btn_file)
        src_row.addWidget(btn_dir)
        layout.addLayout(src_row)

        # -- 格式检测提示 ------------------------------------
        self._fmt_label = QLabel("格式: —")
        self._fmt_label.setStyleSheet("color: #858585; font-size: 9pt; margin-left: 2px;")
        layout.addWidget(self._fmt_label)

        # -- 模型名 ------------------------------------------
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("导入后的名称（如 Monika-v4）")
        form.addRow("模型名:", self._name_edit)
        layout.addLayout(form)

        # -- 分隔线 ------------------------------------------
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #3e3e42;")
        layout.addWidget(sep)

        # -- 进度条 ------------------------------------------
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        layout.addWidget(self._progress)

        # -- 日志输出 -----------------------------------------
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setStyleSheet(
            "QTextEdit {"
            "  background: #1a1a1a; color: #cccccc;"
            "  font-family: Consolas, 'Courier New', monospace;"
            "  font-size: 9pt;"
            "  border: 1px solid #3e3e42; border-radius: 4px;"
            "}"
        )
        layout.addWidget(self._log_box, 1)

        # -- 底部按钮 -----------------------------------------
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._import_btn = QPushButton("导入")
        self._import_btn.setFixedWidth(88)
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._on_import_clicked)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setFixedWidth(88)
        self._cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(self._import_btn)
        btn_row.addWidget(self._cancel_btn)
        layout.addLayout(btn_row)

    # -- 内部辅助 ---------------------------------------------

    def _browse_file(self):
        start = self._src_edit.text() or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 GGUF 模型文件", start,
            "GGUF 模型 (*.gguf);;所有文件 (*.*)"
        )
        if path:
            self._src_edit.setText(path)

    def _browse_dir(self):
        start = self._src_edit.text() or ""
        path = QFileDialog.getExistingDirectory(
            self, "选择 HF safetensors 文件夹", start
        )
        if path:
            self._src_edit.setText(path)

    def _on_src_changed(self, text: str):
        from utils.model_importer import detect_format
        fmt = detect_format(text.strip())

        _labels = {
            "gguf":           "格式: ✓ GGUF 文件（直接复制并注册）",
            "hf_safetensors": "格式: ✓ HF safetensors 文件夹（转换为 F16 GGUF）",
            "unknown":        "格式: ✗ 无法识别（需要 .gguf 文件 或含 config.json+*.safetensors 的文件夹）",
        }
        _colors = {
            "gguf":           "#4ec9b0",
            "hf_safetensors": "#ffcc33",
            "unknown":        "#f48771",
        }
        self._fmt_label.setText(_labels.get(fmt, "格式: —"))
        self._fmt_label.setStyleSheet(
            f"color: {_colors.get(fmt, '#858585')}; font-size: 9pt; margin-left: 2px;"
        )

        # 自动填充模型名（若用户尚未手动输入）
        if fmt in ("gguf", "hf_safetensors") and not self._name_edit.text():
            base = os.path.basename(text.strip().rstrip("/\\"))
            self._name_edit.setText(os.path.splitext(base)[0])

        self._import_btn.setEnabled(fmt in ("gguf", "hf_safetensors"))

    def _append_log(self, msg: str):
        self._log_box.append(msg)
        sb = self._log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        if msg:
            self._append_log(msg)

    # -- 导入动作（qasync） ------------------------------------

    @asyncSlot()
    async def _on_import_clicked(self):
        from utils.model_importer import detect_format, import_gguf, import_hf

        src  = self._src_edit.text().strip()
        name = self._name_edit.text().strip()

        if not src or not name:
            self._append_log("错误：请填写源路径和模型名")
            return

        fmt = detect_format(src)
        if fmt == "unknown":
            self._append_log("错误：无法识别源格式")
            return

        # 禁用控件，防止重入
        self._import_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(0)
        self._append_log(f"开始导入: {src}")
        self._append_log(f"目标名称: {name}")

        try:
            if fmt == "gguf":
                await import_gguf(src, name, self._on_progress)
            else:
                await import_hf(src, name, self._on_progress)

            self._append_log("✓ 导入成功，模型已加入注册表")
            self.imported_model_name = name

            # 变为"完成"按钮，点击关闭并返回 Accepted
            self._import_btn.setText("完成")
            self._import_btn.setEnabled(True)
            self._import_btn.clicked.disconnect()
            self._import_btn.clicked.connect(self.accept)
            self._cancel_btn.setEnabled(True)

        except Exception as exc:
            self._append_log(f"✗ 导入失败: {exc}")
            self._progress.setValue(0)
            self._import_btn.setEnabled(True)
            self._cancel_btn.setEnabled(True)
