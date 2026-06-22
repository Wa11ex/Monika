from datetime import datetime
from utils.config_loader import get_config

class Tools:
    def __init__(self):
        pass
    
    def get_current_time_str(self):
        '''获取当前时间的格式化字符串'''
        try:
            time_format = get_config("context.time.format", "%Y/%m/%d 周一 %H:%M")
            current_time = datetime.now()
            weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][current_time.weekday()] # 翻译

            if "%A" in time_format:
                # 暂时用占位符替换 %A，进行时间格式化
                temp_format = time_format.replace("%A", "__WEEKDAY__")
                time_str = current_time.strftime(temp_format)
                time_str = time_str.replace("__WEEKDAY__", weekday_cn)
            else:
                time_str = current_time.strftime(time_format)
            return time_str
        except Exception as e:
            print(f"!!! [Tools] 获取时间字符串失败: {e}")
            return datetime.now().strftime("%Y-%m-%d %H:%M")