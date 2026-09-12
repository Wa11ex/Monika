'''
系统工具 — 始终可用的基础功能（时间、系统信息等）

与 file_tools / web_tools 不同，这里的函数**不通过 Tool Calling 机制调用**，
而是在每次 LLM 请求组装时直接注入 context，因此无需继承 BaseTool
'''

from datetime import datetime
from utils.config_loader import get_config


def get_current_time_str() -> str:
    '''获取当前时间的格式化字符串，用于注入 LLM context

    格式由 config.yaml 中 context.time.format 控制（默认含中文星期）
    '''
    try:
        time_format = get_config("context.time.format", "%Y/%m/%d 周一 %H:%M")
        current_time = datetime.now()
        weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][current_time.weekday()]
        if "%A" in time_format:
            temp_format = time_format.replace("%A", "__WEEKDAY__")
            time_str = current_time.strftime(temp_format)
            time_str = time_str.replace("__WEEKDAY__", weekday_cn)
        else:
            time_str = current_time.strftime(time_format)
        return time_str
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")
