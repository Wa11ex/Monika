'''音频设备工具 - 设备查找与信息查询'''

import sounddevice as sd


def find_device(keyword: str) -> int | None:
    '''关键字查找输入设备，返回设备ID，未找到返回 None'''
    keyword_lower = keyword.lower()
    for i, d in enumerate(sd.query_devices()):
        if keyword_lower in d['name'].lower() and d['max_input_channels'] > 0:
            return i
    return None


def get_device_channels(device_id: int) -> int:
    '''返回最大输入通道数'''
    return sd.query_devices(device_id)['max_input_channels']


def print_all_devices():
    '''print_all_devices'''
    print("所有音频设备:")
    for i, d in enumerate(sd.query_devices()):
        ch_in  = d['max_input_channels']
        ch_out = d['max_output_channels']
        if ch_in > 0:
            print(f"  [{i:2d}] IN  {d['name']}  ({ch_in}ch)")
        if ch_out > 0:
            print(f"  [{i:2d}] OUT {d['name']}  ({ch_out}ch)")
            
if __name__ == "__main__":
    print_all_devices()
