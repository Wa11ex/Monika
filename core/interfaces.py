'''
Monika-V3 核心接口定义

所有组件的抽象基类/协议，用于解耦和预留扩展空间，尝试一下
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional


# ===================== 模型能力声明 =====================

@dataclass
class ModelCapabilities:
    '''描述当前加载模型的能力集合
    
    由各 Brain 子类的 get_capabilities() 方法返回，供 UI 层用于防呆控制
    所有字段均提供安全的默认值（最保守状态），子类若无法探测则直接返回默认对象
    '''
    backend: str = "unknown"           # "gguf" | "ollama" | "unknown"
    model_name: str = "unknown"        # 模型文件名或模型标识符
    architecture: str = "unknown"      # 模型架构（如 qwen2, llama, llava 等）
    supports_vision: bool = False      # 是否支持图像/视觉输入
    supports_native_tools: bool = False  # 是否原生支持 Function Calling / Tool Use
    supports_thinking: bool = False      # 是否支持显式推理链（<think> 模式，如 Qwen3.5）
    max_context_length: int = 4096     # 最大上下文长度（token 数）


# ===================== LLM 大脑接口 =====================

class BaseBrain(ABC):
    '''LLM 推理后端的统一接口'''

    @abstractmethod
    async def think_stream(self, text: str, speaker_info: Dict = None) -> AsyncGenerator[str, None]:
        '''
        流式生成回复

        Args:
            text: 用户输入文本
            speaker_info: 说话人信息 (由说话人识别系统提供，可选)

        Yields:
            逐字符/逐 token 的回复片段
        '''
        ...

    def get_history(self) -> List[Dict]:
        '''获取当前对话历史（子类可选覆盖）'''
        return []

    def clear_history(self):
        '''清空对话历史（子类可选覆盖）'''
        pass

    def get_capabilities(self) -> ModelCapabilities:
        '''返回当前模型的能力描述（可选实现，默认返回保守的全禁用状态）'''
        return ModelCapabilities()


# ===================== 语音识别接口 =====================

class BaseEars(ABC):
    '''语音输入模块的统一接口'''

    @abstractmethod
    def listen_auto(self) -> Optional[str]:
        '''
        自动录音+识别，返回文本
        返回 "exit" 表示用户要求退出，None 表示无有效输入
        '''
        ...

    @abstractmethod
    def transcribe(self, audio_data) -> Optional[str]:
        '''将音频数据转录为文本'''
        ...

    def start_interrupt_monitoring(self):
        '''开启打断检测'''
        pass

    def stop_interrupt_monitoring(self):
        '''停止打断检测'''
        pass

    def check_interrupt(self) -> bool:
        '''检查是否检测到打断'''
        return False

    def identify_speaker(self, audio_data) -> Optional[Dict]:
        '''
        声纹识别：识别说话人身份

        Args:
            audio_data: 音频数据 (float32 mono ndarray)

        Returns:
            说话人信息字典 {"name": ..., "is_new": bool, "confidence": float} 或 None
        '''
        return None


# ===================== 记忆系统接口 =====================

class BaseMemory(ABC):
    '''长期记忆管理的统一接口'''

    @abstractmethod
    def add_conversation(self, user_msg: str, assistant_msg: str, metadata: Dict = None):
        '''存储一轮对话'''
        ...

    @abstractmethod
    def retrieve_relevant(self, query: str, n: int = 3) -> List[Dict]:
        '''检索相关历史记忆'''
        ...

    @abstractmethod
    def get_memory_context(self, current_query: str, max_memories: int = 3) -> str:
        '''生成注入 LLM 的记忆上下文字符串'''
        ...

    def update_profile(self, key: str, value):
        '''更新用户画像（支持点号分隔的嵌套键，如 'preferences.coffee'）'''
        pass

    def add_important_fact(self, fact: str):
        '''添加重要事实到用户画像'''
        pass

    def get_stats(self) -> Dict: 
        '''获取记忆统计信息'''
        return {}


# ===================== TTS 语音合成接口 =====================

class BaseTTS(ABC):
    '''文本转语音的统一接口'''

    @abstractmethod
    async def synthesize(self, text: str) -> Optional[bytes]:
        '''
        将文本合成为 WAV 音频字节

        Args:
            text: 要合成的文本

        Returns:
            WAV 格式的音频字节，失败返回 None
        '''
        ...

    async def cancel(self):
        '''取消正在进行的合成'''
        pass


# ===================== 感知输入接口 =====================

class BasePerception(ABC):
    '''用户输入感知模块的统一接口（麦克风/键盘/其他）'''

    @abstractmethod
    async def get_next_event(self) -> Dict:
        '''
        异步获取下一个用户事件

        Returns:
            事件字典，包含以下字段：
            - type: "text" | "silence" | "exit"
            - data: str (事件关联的文本，type=silence/exit 时可为空)
            - source: str (事件来源，如 "ears" / "keyboard")
            - timestamp: float (Unix 时间戳)
            - po_count: int (可选，PO 触发次数，type=silence 时提供)
        '''
        ...


# ===================== 虚拟形象控制接口 =====================

class BaseAvatar(ABC):
    '''虚拟形象（VTube Studio 等）的统一接口'''

    @abstractmethod
    async def connect(self) -> bool:
        '''连接到虚拟形象控制端'''
        ...

    @abstractmethod
    async def set_expression(self, name: str):
        '''设置表情'''
        ...

    @abstractmethod
    async def trigger_motion(self, name: str):
        '''触发动作'''
        ...

    @abstractmethod
    async def close(self):
        '''关闭连接'''
        ...


# ===================== 工具调用接口 =====================

class BaseTool(ABC):
    '''可被 LLM 调用的外部工具'''

    @property
    @abstractmethod
    def name(self) -> str:
        '''工具名称'''
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        '''工具描述（会注入到 LLM prompt 中）'''
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict:
        '''参数的 JSON Schema 描述'''
        ...

    @abstractmethod
    async def execute(self, params: Dict) -> Any:
        '''
        执行工具调用

        Args:
            params: 调用参数

        Returns:
            ToolResult（推荐）或工具执行结果的文本描述
        '''
        ...

