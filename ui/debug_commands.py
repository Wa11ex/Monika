'''
Debug 命令注册表，忽略\之后的内容
== 已有命令 ==
    \check --sound   切换 AIUEO 口型调试面板
    \tool --list     列出已注册的工具
    \tool --test     测试工具调用（read_file('README.md')）
    \tool --pwd      显示当前工作目录（工具只能访问此目录下的文件）
    \bench           输出 Benchmark 报告
    \bench --json    输出 Benchmark JSON 原始数据
'''

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.main_window import MainWindow


# -- 命令处理函数 --------------------------------------------------------------

def _cmd_check_sound(win: "MainWindow") -> None:
    '''\check --sound'''
    win._aiueo_panel.setVisible(not win._aiueo_panel.isVisible())


def _cmd_tool_list(win: "MainWindow") -> None:
    '''\tool --list'''
    try:
        from core.tool_registry import get_tool_registry
        registry = get_tool_registry()
        if registry.count == 0:
            win._log("[Tool] 没有已注册的工具请在 main.py 中注册")
        else:
            win._log(f"[Tool] 已注册 {registry.count} 个工具:")
            for name in registry.list_names():
                tool = registry.get(name)
                perm = tool.permission.value if tool else "?"
                win._log(f"  - {name} (permission: {perm})")
    except Exception as e:
        win._log(f"[Tool] 获取工具列表失败: {e}")


def _cmd_tool_test(win: "MainWindow") -> None:
    '''\tool --test'''
    import asyncio
    try:
        from core.tool_registry import get_tool_registry
        registry = get_tool_registry()
        if registry.count == 0:
            win._log("[Tool] 没有已注册的工具，无法测试")
            return
        win._log("[Tool] 运行测试: read_file('README.md')...")

        async def _test():
            result = await registry.execute("read_file", {"path": "README.md"})
            if result.success:
                win._log(f"[Tool] read_file 成功: {len(result.content)} 字符")
                win._log(f"[Tool] 前 100 字符: {result.content[:100]}...")
            else:
                win._log(f"[Tool] [FAIL] read_file 失败: {result.error}")

        # 在 qasync 事件循环中调度
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(_test())
    except Exception as e:
        win._log(f"[Tool] 测试异常: {e}")


def _cmd_tool_pwd(win: "MainWindow") -> None:
    '''\tool --pwd'''
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    win._log(f"[Tool] 项目根目录: {root}")
    win._log(f"[Tool] 文件工具只能操作此目录下的文件")
    # 列出根目录内容
    try:
        items = sorted(os.listdir(root))
        dirs = [f"  {d}/" for d in items if os.path.isdir(os.path.join(root, d))]
        files = [f"  {f}" for f in items if os.path.isfile(os.path.join(root, f))]
        win._log(f"[Tool] 目录: {len(dirs)} 个, 文件: {len(files)} 个")
        if dirs:
            win._log("[Tool] 子目录:\n" + "\n".join(dirs[:10]))
    except Exception as e:
        win._log(f"[Tool] 列出失败: {e}")


# -- 命令注册表（可编辑区域） ---------------------------------------------------
def _cmd_bench_report(win: "MainWindow") -> None:
    '''\bench'''
    try:
        from core.benchmark import bench
        report = bench.report()
        for line in report.split("\n"):
            win._log(line)
    except Exception as e:
        win._log(f"[Benchmark] 获取报告失败: {e}")


def _cmd_bench_json(win: "MainWindow") -> None:
    '''\bench --json'''
    try:
        from core.benchmark import bench
        win._log(bench.to_json())
    except Exception as e:
        win._log(f"[Benchmark] 导出失败: {e}")

COMMANDS: dict[str, callable] = {
    "check --sound": _cmd_check_sound,
    "tool --list": _cmd_tool_list,
    "tool --test": _cmd_tool_test,
    "tool --pwd": _cmd_tool_pwd,
    "bench": _cmd_bench_report,
    "bench --json": _cmd_bench_json,
}


# -- 分发器（供 main_window 调用） ---------------------------------------------

def dispatch(win: "MainWindow", text: str) -> None:
    '''检索和执行命令'''
    cmd_key = text[1:]  # 去掉开头的 \
    handler = COMMANDS.get(cmd_key)
    if handler is not None:
        try:
            handler(win)
        except Exception as e:
            win._log(f"[Debug] 命令执行出错: {e}")
    else:
        win._log(f"[Debug] 命令未找到: {text}")
        # 列出所有可用命令作为提示
        available = "; \n".join(f"\\{k}" for k in COMMANDS)
        win._log(f"[Debug] 可用命令: \n{available}")
