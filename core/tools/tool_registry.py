'''
Tool Registry — 管理所有已注册的工具，提供定义导出和执行
'''

from typing import Any, Dict, List, Optional
from .base import BaseTool, ToolResult
from utils.logger import get_logger

log = get_logger(__name__)


class ToolRegistry:
    '''工具注册中心

    单例模式：整个应用共用一个实例
    '''

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    # -- 注册 / 查找 ----------------------------------------

    def register(self, tool: BaseTool) -> None:
        '''注册一个工具（同名覆盖）'''
        self._tools[tool.name] = tool
        log.debug("[ToolRegistry] 已注册: %s", tool.name)

    def register_all(self, tools: List[BaseTool]) -> None:
        '''批量注册工具'''
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[BaseTool]:
        '''按名称查找工具'''
        return self._tools.get(name)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        return len(self._tools)

    # -- 定义导出 --------------------------------------------

    def get_definitions(self) -> List[Dict[str, Any]]:
        '''导出所有工具的 OpenAI-compatible function definitions

        这个列表直接传给 Jinja 模板的 `tools` 变量
        '''
        return [tool.to_definition() for tool in self._tools.values()]

    # -- 执行 ------------------------------------------------

    async def execute(self, name: str, params: Dict[str, Any]) -> ToolResult:
        '''按名称执行工具

        Args:
            name:   工具名称（function name）
            params: 参数字典

        Returns:
            ToolResult — 成功/失败
        '''
        tool = self._tools.get(name)
        if tool is None:
            known = ", ".join(self._tools.keys()) or "(none)"
            return ToolResult(False, "", f"未知工具 '{name}'可用: {known}")
        try:
            return await tool.execute(params)
        except Exception as e:
            log.error("[ToolRegistry] 工具 '%s' 执行异常: %s", name, e)
            return ToolResult(False, "", f"工具执行异常: {e}")


# -- 全局单例 ------------------------------------------------

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    '''获取全局 ToolRegistry 单例'''
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


