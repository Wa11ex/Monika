'''
Tool Format Adapter — 用于适配不同工具格式
'''

from core.brain_state import (
    ToolFormatAdapter,
    QwenXMLAdapter,
    create_tool_adapter,
    _detect_tool_format,
)

# 旧名称
create_adapter = create_tool_adapter
_detect_format = _detect_tool_format

__all__ = [
    "ToolFormatAdapter",
    "QwenXMLAdapter",
    "create_adapter",
    "_detect_format",
]
