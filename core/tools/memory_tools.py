'''
记忆相关工具 — 用户画像、对话反思等

RememberAboutUser: 模型在对话中了解到用户信息时主动记录
'''

from __future__ import annotations

import json
import os
from typing import Any, Dict

from core.tools.base import BaseTool, ToolPermission, ToolResult
from utils.config_loader import get_config
from utils.logger import get_logger

log = get_logger(__name__)


class RememberAboutUser(BaseTool):
    '''记录关于用户的一条事实或偏好'''

    name = "remember_about_user"
    permission = ToolPermission.AUTO

    @property
    def description(self) -> str:
        return (
            "记录一条关于用户的事实、偏好或习惯"
            "当你从对话中了解到用户的重要信息时调用，"
            "例如'用户喜欢咖啡'、'用户是程序员'、'用户最近在学日语'"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "关于用户的事实，用简短的句子描述",
                },
            },
            "required": ["fact"],
        }

    @property
    def usage_guide(self) -> str:
        return (
            "当你在对话中了解到用户的新信息时（如职业、喜好、习惯、近期活动），"
            "使用此工具记录下来这样下次见面时你就能记得关于 TA 的事"
            "例如：用户说'我昨天熬夜写代码'→ remember_about_user(fact='用户经常熬夜写代码')"
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        fact = params.get("fact", "").strip()
        if not fact:
            return ToolResult(success=False, error="事实内容不能为空")

        try:
            from utils.config_loader import get_config
            profile_path = get_config("memory.profile.path", "./assets/memory/chr.json")
            if not os.path.isabs(profile_path):
                from utils.helpers import _get_project_root
                profile_path = os.path.join(_get_project_root(), profile_path)

            # 读取现有画像
            profile = {}
            if os.path.isfile(profile_path):
                try:
                    with open(profile_path, "r", encoding="utf-8") as f:
                        profile = json.load(f)
                except (json.JSONDecodeError, IOError):
                    profile = {}

            # 确保字段存在
            if "important_facts" not in profile:
                profile["important_facts"] = []
            if "preferences" not in profile:
                profile["preferences"] = {}

            # 去重
            if fact not in profile["important_facts"]:
                profile["important_facts"].append(fact)

            # 限制数量（保留最近 50 条）
            if len(profile["important_facts"]) > 50:
                profile["important_facts"] = profile["important_facts"][-50:]

            from datetime import datetime, timezone
            profile["last_updated"] = datetime.now(timezone.utc).isoformat()

            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)

            log.info("[MemoryTool] 记录用户事实: %s", fact[:80])
            return ToolResult(
                success=True,
                content=f"已记录: {fact}（当前共 {len(profile['important_facts'])} 条）",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"记录失败: {e}")
