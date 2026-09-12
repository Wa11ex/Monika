import asyncio
import os
import sys
import traceback

# -- 强制使用 UTF-8 编码 --
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logging, get_logger
setup_logging(
    level="INFO",
    log_dir="logs"
)
log = get_logger(__name__)

from utils import load_config, get_config
from utils.helpers import resolve_path
from core.launcher import launch_all, shutdown_all
from core import MonikaBus
from core import VTSManager
from core import Ears
from utils.benchmark import bench
from core import MicPerception, KeyboardPerception
from core.event import PoStateMachine

PO_TRIGGER_TEXT = "[沉默]"


def _create_brain(memory_manager=None):
    '''选择 LLM 后端'''
    backend = get_config("brain.backend", "ollama")
    if backend == "gguf":
        from core.brain_loader import GGufBrain as Brain
        model_path = get_config("brain.gguf.model_path", "assets/model/Monika-v3.gguf")
        if not os.path.isabs(model_path):
            model_path = resolve_path(model_path)
        return Brain(model_path=model_path, memory_manager=memory_manager)
    else:
        from core.brain_ollama import Brain
        model_name = get_config("brain.ollama.model_name", "Monika:latest")
        return Brain(model_name=model_name, memory_manager=memory_manager)


async def init_monika(log_callback=None, progress_callback=None):
    '''初始化并返回莫妮卡的核心组件字典，供给 GUI 或 Main 调用
        log_callback: 日志回调函数，签名为 log_callback(message: str)，可以没有
        progress_callback: 进度回调，签名 progress_callback(stage: str, pct: int)
            stage: 当前阶段名称，pct: 0~100 进度百分比
    '''
    def _progress(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)

    def _log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    
    load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    
    # 如果有日志回调，重定向到回调
    if log_callback:
        from core.launcher import set_log_callback
        set_log_callback(log_callback)
    
    _progress("启动 TTS 服务...", 5)
    tts_task = await launch_all() # tts流最后开始
    _log(">>> [Main] 正在初始化莫妮卡核心组件...")
    _progress("加载核心组件...", 15)

    vts = VTSManager()
    bus = MonikaBus(vts)
    ears_device = get_config("ears.device", "cpu")

    # -- 并行加载两条独立链 --------------------------------------------
    # 链 A: SpeakerIdentifier -- Ears
    # 链 B: MemoryManager -- Brain 这就是独立绑定
    # 用 asyncio.gather 真并行

    async def _load_ears_chain():
        '''SpeakerIdentifier || ASR 并行加载'''
        sp_id = None

        async def _load_speaker_id():
            nonlocal sp_id
            if not get_config("speaker_id.enabled", False):
                return
            try:
                from core.audio.speaker_id import SpeakerIdentifier
                sp_data_path = get_config("speaker_id.data_path", "memory/data/speakers.json")
                if not isinstance(sp_data_path, str):
                    sp_data_path = "memory/data/speakers.json"
                _sp_path = (
                    os.path.join(PROJECT_ROOT, sp_data_path)
                    if not os.path.isabs(sp_data_path) else sp_data_path
                )
                _sp_thresh = float(get_config("speaker_id.threshold", 0.75))
                _log(">>> [Main] 正在加载声纹识别模型...")
                sp_id = await asyncio.to_thread(
                    lambda: SpeakerIdentifier(data_path=_sp_path, threshold=_sp_thresh)
                )
            except Exception as e:
                _log(f"!!! [Main] 声纹识别加载失败: {e}")

        async def _load_asr():
            _log(">>> [Main] 正在加载 ASR 模型...")
            return await asyncio.to_thread(
                lambda: Ears(device=ears_device, speaker_identifier=sp_id)
            )

        # SpeakerID 和 ASR 并行加载
        _sp_task = asyncio.create_task(_load_speaker_id())
        _ears_task = asyncio.create_task(_load_asr())
        _ears = await _ears_task
        await _sp_task
        # 并行加载期间 sp_id 可能 None，完成后补绑定
        if sp_id is not None:
            _ears.speaker_id = sp_id
        return _ears, sp_id

    async def _load_brain_chain():
        '''MemoryManager || Brain 并行加载（Brain 初始化不依赖 MemoryManager 完成）'''
        mem = None

        async def _load_memory():
            nonlocal mem
            try:
                from memory import MemoryManager
                memory_dir = get_config("memory.memory_dir", "./memory")
                if not isinstance(memory_dir, str):
                    memory_dir = "./memory"
                _log(">>> [Main] 正在加载记忆系统...")

                def _load_mem(_dir=memory_dir):
                    _m = MemoryManager(memory_dir=_dir)
                    if get_config("memory.lore.enabled", False):
                        lore_path = get_config("memory.lore.chunks_path", "assets/lore_chunks.json")
                        if not os.path.isabs(lore_path):
                            lore_path = os.path.join(PROJECT_ROOT, lore_path)
                        _m.seed_lore(lore_path)
                    else:
                        log.info("[Main] memory.lore.enabled=false，跳过 Lore 记忆写入")
                    return _m

                mem = await asyncio.to_thread(_load_mem)
            except Exception as e:
                _log(f"!!! [Main] 记忆系统加载失败: {e}")

        async def _load_brain():
            _log(">>> [Main] 正在加载语言模型...")
            return await asyncio.to_thread(lambda: _create_brain(memory_manager=mem))

        # MemoryManager 和 Brain 并行加载
        _mem_task = asyncio.create_task(_load_memory())
        _brain_task = asyncio.create_task(_load_brain())
        _brain = await _brain_task
        await _mem_task
        # Brain 初始化时 mem 可能还 None，完成后补绑定
        if mem is not None:
            _brain.memory = mem

        try:
            # 兼容性探针能力，输出日志
            caps = _brain.get_capabilities()
            if asyncio.iscoroutine(caps):
                caps = await caps
            log.info(
                "[Brain] 模型能力探针: backend=%s | model=%s |\n arch=%s | "
                "vision=%s |\n native_tools=%s | thinking=%s |\n max_ctx=%d",
                caps.backend, caps.model_name, caps.architecture,
                caps.supports_vision, caps.supports_native_tools,
                caps.supports_thinking, caps.max_context_length,
            )
        except Exception as e:
            log.warning("[Brain] 能力探针异常: %s", e)

        return _brain, mem

    # -- 三路并行：TTS health-wait，Ears，Brain(GGUF) --
    # 预计能少个30s的等待时间？
    async def _await_tts(task):
        # tts_task 由 launch_all() 返回，进程已启动但健康检查尚未完成的时候是None
        if task is None:
            return True
        ok = await task
        if not ok:
            log.warning("[Main] TTS 服务未能就绪，语音合成功能可能不可用")
        return ok

    # 工具注册也放入并行加载
    async def _register_tools():
        if not get_config("tools.enabled", False):
            return
        try:
            from core.tools.tool_registry import get_tool_registry
            from core.tools.file_tools import ReadFile, WriteFile, ListDirectory, SearchFiles
            from core.tools.web_tools import WebSearch
            from core.tools.memory_tools import RememberAboutUser
            registry = get_tool_registry()
            registry.register_all([
                ReadFile(), WriteFile(), ListDirectory(), SearchFiles(),
                WebSearch(), RememberAboutUser(),
            ])
            _log(f">>> [Main] 已注册 {registry.count} 个工具")
        except Exception as e:
            _log(f"!!! [Main] 工具初始化失败: {e}")

    (ears, speaker_identifier), (brain, memory), _, _ = await asyncio.gather(
        _load_ears_chain(),
        _load_brain_chain(),
        _await_tts(tts_task),
        _register_tools(),
    )
    
    _progress("核心组件就绪", 80)

    # -- RL 信号收集器 --
    try:
        from rl.collector import SignalCollector
        brain.collector = SignalCollector(log_dir=get_config("rl.log_dir", "rl/logs"))
        log.info("[Main] RL 信号收集器已启动")
    except Exception as e:
        log.warning("[Main] RL 信号收集器初始化失败: %s", e)

    _progress("收尾中...", 90)

    # -- ASR 后台预加载（利用 TTS 加载的等待时间并行加载） --
    async def _preload_asr():
        try:
            await asyncio.to_thread(ears._ensure_model)
            log.info("[Main] ASR 预加载完成")
        except Exception as e:
            log.warning("[Main] ASR 预加载失败（将在首次使用时重试）: %s", e)
    asyncio.create_task(_preload_asr())

    # -- PO 与 Input 模式 --
    po_cfg = get_config("proactive_output", {})
    po = PoStateMachine.from_config(po_cfg) if isinstance(po_cfg, dict) and po_cfg.get("enabled", True) else None
    
    audio_input_mode = get_config("audio.input_mode", "mic_always_on")

    if audio_input_mode == "typing":
        perception = KeyboardPerception()
    else:
        perception = MicPerception(ears=ears, po=po)

    _progress("初始化完成", 100)
    return {
        "vts": vts,
        "bus": bus,
        "ears": ears,
        "memory": memory,
        "speaker_identifier": speaker_identifier,
        "brain": brain,
        "po": po,
        "input_mode": audio_input_mode,
        "perception": perception,
    }

async def shutdown_monika(components):
    '''一个一个关闭
    components: init_monika 的组件字典
    '''
    log.info("[Main] 开始安全关闭流程...")
    ears = components.get("ears")
    if ears: ears.stop_interrupt_monitoring()
    
    bus = components.get("bus")
    if bus: await bus.shutdown()
    
    vts = components.get("vts")
    if vts: await vts.close()
    
    memory = components.get("memory")
    if memory:
        try: memory.shutdown()
        except: pass
        
    speaker_identifier = components.get("speaker_identifier")
    if speaker_identifier:
        try: speaker_identifier.shutdown()
        except: pass
    
    shutdown_all()
    log.info("[Main] 程序已完全关闭")


async def main():
    components = await init_monika()

    # -- 提升进程优先级（ Windows 独享）----------------------------
    if sys.platform == "win32":
        import psutil
        try:
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except Exception:
            pass

    vts = components["vts"]
    bus = components["bus"]
    ears = components["ears"]
    brain = components["brain"]
    input_mode = components["input_mode"]
    perception = components["perception"]

    log.info("[Main] MONIKA-V3 CORE ONLINE | 模式: %s | 按 ESC 退出", input_mode)

    try:
        while True:
            # -- 0. 定期检查是否需要恢复表情（超时2秒） --
            await bus.check_and_restore_expression_if_timeout(timeout_sec=2.0)

            # -- 1. 通过 Perception get下一个事件 --
            event = await perception.get_next_event()

            event_type = event.get("type")
            user_text = event.get("data")

            # -- 2. 事件分类 --
            if event_type == "exit":
                log.info("[Main] 接收到退出信号，拜拜！")
                break
            elif event_type == "silence":
                # PO 触发
                count = event.get("po_count", 0)
                log.info("[Main] 主动发言触发（第 %d 次冷却）", count)
                user_text = PO_TRIGGER_TEXT # 沉默文本模板
            elif event_type == "text":
                if not user_text or not user_text.strip(): # 空文本不处理
                    continue
            else: # 兜底
                continue

            # 声纹识别结果
            speaker = getattr(ears, "last_speaker", None) if input_mode != "typing" else None
            if speaker:
                label = f"新朋友/{speaker['name']}" if speaker["is_new"] else speaker["name"]
                log.info("[User/%s] %s", label, user_text)
            elif input_mode != "typing":
                log.info("[User] %s", user_text)

            # -- 3. 开启打断与思考 --
            bus.reset_interrupt()
            if input_mode != "typing": # TODO
                ears.start_interrupt_monitoring()

            async def _watch_interrupt():
                while not bus.is_interrupted and not bus.player.has_started_playing:
                    await asyncio.sleep(0.05)
                while not bus.is_interrupted:
                    if ears.check_interrupt():
                        log.info("[Main] 检测到打断信号!")
                        await bus.interrupt()
                        return
                    await asyncio.sleep(0.05)

            watcher = None
            if input_mode != "typing":
                # 初始化监听者
                watcher = asyncio.create_task(_watch_interrupt())

            stream = brain.think_stream(user_text, speaker_info=speaker)
            raw_response = await bus.process_stream_input(stream)

            if not bus.is_interrupted:
                await bus.wait_for_completion()

            # 清理打断监听，卸磨杀驴
            if watcher:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
                ears.stop_interrupt_monitoring()

            brain.commit_response(raw_response, interrupted=bus.is_interrupted)
            if not bus.is_interrupted:
                await vts.set_expression("normal")

    except KeyboardInterrupt:
        log.info("[Main] Ctrl+C 收到，正在关闭...")
    except Exception as e:
        log.exception("[Main] 运行时报错: %s", e)
    finally:
        if getattr(brain, 'collector', None):
            try:
                brain.collector.on_session_end()
            except Exception:
                pass
        await shutdown_monika(components)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("[Main] 再次收到 Ctrl+C，强制退出...")
        shutdown_all()