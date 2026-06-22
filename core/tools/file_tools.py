'''
文件操作工具 — ReadFile, WriteFile, ListDirectory, SearchFiles
'''

import os
import fnmatch
import asyncio
from typing import Any, Dict

from core.tools.base import BaseTool, ToolResult, ToolPermission

# 安全限制：工具只允许操作项目根目录内的文件
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _safe_path(requested: str) -> str | None:
    '''将请求路径规范化并检查是否在项目根目录内返回绝对路径或 None'''
    if not requested or not isinstance(requested, str):
        return None
    # 支持相对路径和绝对路径
    if os.path.isabs(requested):
        resolved = os.path.normpath(requested)
    else:
        resolved = os.path.normpath(os.path.join(_PROJECT_ROOT, requested))
    # 安全检查：不允许跳出项目目录
    if not resolved.startswith(os.path.normpath(_PROJECT_ROOT)):
        return None
    return resolved


class ReadFile(BaseTool):
    '''读取文件内容'''

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. "
            "Use this to examine existing code, configuration, or documentation. "
            "The path is relative to the project root."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root (e.g., 'config.yaml', 'core/brain_base.py')",
                },
            },
            "required": ["path"],
        }

    @property
    def usage_guide(self) -> str:
        return "当用户要求查看、阅读、读取某个文件的内容时使用例如'帮我看看 README'、'读一下 config.yaml'"

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.AUTO

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", "")
        safe = _safe_path(path)
        if safe is None:
            return ToolResult(False, "", f"路径越界或无效: {path}")
        if not os.path.isfile(safe):
            return ToolResult(False, "", f"文件不存在: {path}")
        try:
            content = await asyncio.to_thread(_read_file, safe)
            # 截断过长文件
            if len(content) > 10000:
                content = content[:10000] + f"\n\n... (truncated, total {len(content)} chars)"
            return ToolResult(True, content)
        except Exception as e:
            return ToolResult(False, "", f"读取失败: {e}")


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


class WriteFile(BaseTool):
    '''创建或覆盖文件'''

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Create a new file or overwrite an existing file. "
            "The path is relative to the project root. "
            "IMPORTANT: Always confirm the full content before calling this tool."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root",
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content to write",
                },
            },
            "required": ["path", "content"],
        }

    @property
    def usage_guide(self) -> str:
        return "当用户要求创建、写入、修改、保存文件内容时使用例如'帮我写一个 hello.py'、'把这个存到 config.yaml'注意：此工具会覆盖已有文件"

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.CONFIRM

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", "")
        content = params.get("content", "")
        safe = _safe_path(path)
        if safe is None:
            return ToolResult(False, "", f"路径越界或无效: {path}")
        if not content:
            return ToolResult(False, "", "content 不能为空")
        try:
            await asyncio.to_thread(_write_file, safe, content)
            return ToolResult(True, f"文件已写入: {path} ({len(content)} 字符)")
        except Exception as e:
            return ToolResult(False, "", f"写入失败: {e}")


def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class ListDirectory(BaseTool):
    '''列出目录内容'''

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "List files and directories at the given path. "
            "Use this to explore the project structure before reading files."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the project root (default: '.' = root)",
                },
            },
            "required": [],
        }

    @property
    def usage_guide(self) -> str:
        return "当用户要求列出、查看目录中有哪些文件时使用例如'看看项目里有什么文件'、'列出当前目录'"

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.AUTO

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        path = params.get("path", ".")
        safe = _safe_path(path)
        if safe is None:
            return ToolResult(False, "", f"路径越界或无效: {path}")
        if not os.path.isdir(safe):
            return ToolResult(False, "", f"目录不存在: {path}")
        try:
            entries = await asyncio.to_thread(os.listdir, safe)
            lines = []
            for name in sorted(entries):
                full = os.path.join(safe, name)
                marker = "/" if os.path.isdir(full) else ""
                lines.append(f"  {name}{marker}")
            return ToolResult(True, "\n".join(lines))
        except Exception as e:
            return ToolResult(False, "", f"列表失败: {e}")


class SearchFiles(BaseTool):
    '''按文件名模糊搜索项目中的文件'''

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def permission(self) -> ToolPermission:
        return ToolPermission.AUTO

    @property
    def description(self) -> str:
        return (
            "在项目中搜索文件支持两种模式："
            "① 模糊匹配（如 'brain' 匹配 brain_loader.py）；"
            "② 扩展名/glob 搜索（如 '*.md' 找所有 markdown、'*.py' 找 Python、'test_*.py' 找测试文件）"
            "递归搜索子目录，自动跳过 __pycache__ 和隐藏目录"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索关键词或 glob 模式关键词如 'brain' 匹配文件名含 brain 的文件；"
                        "glob 模式如 '*.md'（所有 markdown）、'*.py'（所有 Python）、'test_*'（测试文件）"
                        "用户说'找md文件'时用 query='*.md'，说'找配置文件'时用 query='config'或'*.yaml'"
                    ),
                },
                "directory": {
                    "type": "string",
                    "description": "搜索目录（相对于项目根目录，默认 '.' 即整个项目）",
                    "default": ".",
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回数（默认 30）",
                    "default": 30,
                },
            },
            "required": ["query"],
        }

    @property
    def usage_guide(self) -> str:
        return (
            "当你不确定文件的完整路径时，先用此工具搜索"
            "用户说'找md文件'→search_files(query='*.md')；"
            "用户说'看下日志代码'→search_files(query='log')；"
            "用户说'找Python测试'→search_files(query='test_*.py')"
            "找到后用 read_file 读取具体文件"
            "可以用 * 匹配任意字符，如 *.yaml、*.json、README*"
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip()
        if not query:
            return ToolResult(success=False, error="搜索关键词不能为空")

        directory = params.get("directory", ".")
        max_results = min(int(params.get("max_results", 30)), 100)

        safe_dir = _safe_path(directory)
        if safe_dir is None:
            return ToolResult(success=False, error=f"路径越界: {directory}")
        if not os.path.isdir(safe_dir):
            return ToolResult(success=False, error=f"目录不存在: {directory}")

        try:
            results = await asyncio.to_thread(
                _search_files, safe_dir, query, max_results
            )
            if not results:
                return ToolResult(
                    success=True,
                    content=f"在 '{directory}' 中未找到匹配 '{query}' 的文件"
                )
            return ToolResult(success=True, content=_format_search_results(results))
        except Exception as e:
            return ToolResult(success=False, error=f"搜索失败: {e}")


def _search_files(root: str, query: str, max_results: int) -> list[dict]:
    '''递归搜索文件，支持模糊匹配和 glob 模式'''
    results = []
    # 是否包含通配符（glob 模式）
    has_glob = any(c in query for c in ('*', '?', '['))

    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过隐藏目录和常见忽略目录
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith('.')
            and d not in ('__pycache__', 'node_modules', '.git', 'venv', 'env')
        ]

        for fname in filenames:
            if fname.startswith('.'):
                continue

            match = False
            if has_glob:
                match = fnmatch.fnmatch(fname, query)
            else:
                # 模糊匹配：文件名包含 query（大小写不敏感）
                match = query.lower() in fname.lower()

            if match:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, _PROJECT_ROOT)
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0
                results.append({
                    "path": rel_path,
                    "size": size,
                })
                if len(results) >= max_results:
                    return results

    return results


def _format_search_results(results: list[dict]) -> str:
    '''格式化搜索结果'''
    lines = [f"找到 {len(results)} 个文件：\n"]
    for r in results:
        size_str = f"{r['size'] / 1024:.1f}KB" if r['size'] > 1024 else f"{r['size']}B"
        lines.append(f"  {r['path']}  ({size_str})")
    return "\n".join(lines)
