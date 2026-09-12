'''
core.event — 事件、状态与流（横切关注点）

  EventBus      : 轻量事件总线（pub/sub）
  TraitManager  : Monika 内部状态（心情/寂寞/好奇心/好感）
  PoStateMachine: 主动发言（PO）状态机
  emit / drain  : 模型原始 token 输出流（调试面板用）
'''

from .EventBus import EventBus, get_event_bus
from .traits import TraitManager, TraitSnapshot, TriggeredAction
from .po_manager import PoStateMachine, PoWindow
from .stream import emit, drain

__all__ = [
    'EventBus', 'get_event_bus',
    'TraitManager', 'TraitSnapshot', 'TriggeredAction',
    'PoStateMachine', 'PoWindow',
    'emit', 'drain',
]
