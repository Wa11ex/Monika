'''
上网工具 — DuckDuckGo + trafilatura

WebSearch: 搜索互联网并可选提取网页正文
'''

from __future__ import annotations

import asyncio
from typing import Any, Dict

import aiohttp

from core.tools.base import BaseTool, ToolPermission, ToolResult
from utils.logger import get_logger

log = get_logger(__name__)

_MAX_SNIPPET_LENGTH = 200
_MAX_CONTENT_LENGTH = 1500
_SEARCH_TIMEOUT = 10
_FETCH_TIMEOUT = 12


class WebSearch(BaseTool):
    '''DuckDuckGo 网页搜索，可选正文提取'''

    name = "web_search"
    permission = ToolPermission.ALWAYS_ASK

    @property
    def description(self) -> str:
        return "搜索互联网获取最新信息返回结果包含标题、URL 和摘要，可选择提取网页正文。"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词（中英文均可）。",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量上限（默认 3，最大 5）结果越少回复越快。",
                    "default": 3,
                },
                "fetch_content": {
                    "type": "boolean",
                    "description": "是否抓取网页正文（默认 false，仅返回标题+摘要）？",
                    "default": False,
                },
            },
            "required": ["query"],
        }

    @property
    def usage_guide(self) -> str:
        return (
            "当用户询问最新信息、实时数据、新闻、百科知识时调用。"
            "优先用 fetch_content=false 只看标题摘要（快速），"
            "只有用户明确要求'详细内容'时才设 fetch_content=true。"
            "max_results 建议 2-3 条，太多结果会超长被截断。"
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        query = params.get("query", "").strip()
        if not query:
            return ToolResult(success=False, error="搜索关键词不能为空")

        max_results = min(int(params.get("max_results", 3)), 5)
        _fc_raw = str(params.get("fetch_content", "")).lower()
        fetch_content = _fc_raw in ("true", "1", "yes")

        try:
            results = await asyncio.to_thread(
                _search_duckduckgo, query, max_results
            )
        except Exception as e:
            log.warning("[WebSearch] 搜索失败: %s", e)
            return ToolResult(success=False, error=f"搜索失败: {e}")

        if not results:
            return ToolResult(success=True, content=f"未找到与「{query}」相关的结果")

        if fetch_content:
            results = await _fetch_contents(results)

        return ToolResult(success=True, content=_format_results(query, results))


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    '''同步搜索 DuckDuckGo'''
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", "")[:_MAX_SNIPPET_LENGTH],
            })
    return results


async def _fetch_contents(results: list[dict]) -> list[dict]:
    '''异步抓取每个结果的正文内容'''
    import trafilatura

    async def _fetch_one(session: aiohttp.ClientSession, r: dict) -> dict:
        url = r.get("href", "")
        if not url:
            return r
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=_FETCH_TIMEOUT),
                headers={"User-Agent": "Monika/1.0 (AI Assistant)"},
            ) as resp:
                if resp.status != 200:
                    r["content"] = f"[HTTP {resp.status}]"
                    return r
                html = await resp.text()
                # trafilatura 提取正文
                extracted = await asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    include_links=False,
                    include_images=False,
                    include_tables=False,
                )
                r["content"] = (
                    extracted[:_MAX_CONTENT_LENGTH]
                    if extracted
                    else "[无法提取正文]"
                )
        except asyncio.TimeoutError:
            r["content"] = "[抓取超时]"
        except Exception as e:
            r["content"] = f"[抓取失败: {e}]"
        return r

    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_one(session, r) for r in results]
        return await asyncio.gather(*tasks)


def _format_results(query: str, results: list[dict]) -> str:
    '''格式化搜索结果为可读文本'''
    lines = [f"搜索「{query}」的结果（共 {len(results)} 条）：\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        url = r.get("href", "")
        body = r.get("body", "")
        content = r.get("content", "")

        lines.append(f"## {i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if body:
            lines.append(f"   摘要: {body}")
        if content:
            lines.append(f"   正文: {content}")
        lines.append("")

    return "\n".join(lines)
