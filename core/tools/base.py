'''
Tool Calling 系统 — 基础类型
允许自定义工具被 LLM 调用，提供权限控制、使用指南和定义导出
'''

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from core.interfaces import BaseTool as BaseToolInterface


class ToolPermission(Enum):
    '''工具权限级别'''
    AUTO = "auto"            # 自动执行，无需确认（如 read_file）
    CONFIRM = "confirm"      # 弹确认框（如 write_file）
    ALWAYS_ASK = "always_ask"  # 每次询问（如 web_search）


@dataclass
class ToolResult:
    '''工具执行结果'''
    success: bool
    content: str = ""              # 返回给 LLM 的文本
    error: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)  # 额外信息（备份路径等）


class BaseTool(BaseToolInterface):
    '''可被 LLM 调用的自定义工具
    继承自 core.interfaces.BaseTool 接口，增加权限控制、使用指南和定义导出
    子类需实现 name, description, parameters_schema, execute
    '''

    @property
    @abstractmethod
    def name(self) -> str:
        '''工具名称'''
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        '''工具描述'''
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        '''参数的 JSON Schema（OpenAI 格式）'''
        ...

    @property
    def permission(self) -> ToolPermission:
        '''默认权限级别，子类可覆盖'''
        return ToolPermission.CONFIRM

    @property
    def usage_guide(self) -> str:
        '''工具使用指南
        子类应返回中文描述，说明"什么时候该用这个工具"
        例如："当用户要求查看或读取文件内容时，使用此工具"
        '''
        return ""

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        '''执行工具调用'''
        ...

    def to_definition(self) -> Dict[str, Any]:
        '''生成 OpenAI-compatible function definition'''
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }
