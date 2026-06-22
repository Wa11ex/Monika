'''
UI 工具函数集合
'''

import re
from PyQt6.QtWidgets import QWidget, QScrollArea


_META_PAT = re.compile(r'[\(\（][^\)\）]*[\)\）]|\[[^\]]*\]|【[^】]*】')

def strip_metadata(text: str) -> str:
    '''过滤： (action), [tag], 【tag】'''
    return re.sub(_META_PAT, '', text).strip()

def scrolled(widget: QWidget) -> QScrollArea:
    '''滚动条'''
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    return area
