import asyncio
import sounddevice as sd
import numpy as np
import threading
import queue
import re
import io
import soundfile as sf
import os
import time
from core.vts_client import VTSManager
from core.interfaces import BaseTTS
from utils.config_loader import get_config
from utils.logger import get_logger
from core.benchmark import bench

log = get_logger(__name__)

_lipsync = None

def _get_lipsync():
    '''惰性加载 lipsync 模块，这个是为了避免启动时拉入numpy和scipy这两个大库'''
    global _lipsync
    if _lipsync is None:
        from core.audio import lipsync as _lipsync
    return _lipsync


class AudioPlayer(threading.Thread):
    def __init__(self, device_id, loop):
        super().__init__()
        self.queue = queue.Queue()
        self.device_id = device_id
        self.daemon = True
        self._stop_flag = False
        self.loop = loop
        
        # 使用 asyncio.Event 进行事件循环的安全交互
        self.finished_event_async = asyncio.Event()
        self._skip_current = False
        self._halted = False  # 阻止信号，给打断用的
        self.has_started_playing = False  # 第一块音频开始输出后为 True，供打断守卫使用

    def set_finished_async(self):
        '''safe set'''
        self.loop.call_soon_threadsafe(lambda: self.finished_event_async.set())

    def clear_finished_async(self):
        '''safe clear'''
        self.loop.call_soon_threadsafe(lambda: self.finished_event_async.clear())

    def add_wav(self, wav_bytes):
        '''音频字节流'''
        if self._halted:# 阻止信号
            return
        self.queue.put(wav_bytes)

    def run(self):
        log.info("[AudioPlayer] 线程启动，设备ID: %s", self.device_id)
        self.set_finished_async()  # 一开始处于空闲完毕状态
        while not self._stop_flag:
            try:
                wav_bytes = self.queue.get(timeout=0.1)
                if wav_bytes is None: break

                
                if self._halted:
                    self.queue.task_done() # 打断后丢弃队列中残留的音频
                    continue

                # self.clear_finished_async()  # 由生产者提前 clear 保证无缝衔接

                try:
                    data, sr = sf.read(io.BytesIO(wav_bytes))
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    # 统一为 (N, channels) 形状
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    channels = data.shape[1]

                    # 两个用来记录的流
                    pos_ref    = [0]
                    _first_cb  = [True]   # 第一次 callback 触发，声音真正开始输出

                    def callback(outdata, frames, _time_info, _status):
                        # 第一帧回调时标记真实出声，lipsync 以此为时间零点开始同步
                        if _first_cb[0]:
                            _first_cb[0] = False
                            self.has_started_playing = True
                        
                        # 检测到打断：写入静音并立即中止流（Pa_AbortStream）
                        if self._skip_current or self._halted:
                            outdata[:] = 0
                            raise sd.CallbackAbort
                        remaining = len(data) - pos_ref[0]
                        if remaining <= 0:
                            raise sd.CallbackStop
                        chunk = min(frames, remaining)
                        outdata[:chunk] = data[pos_ref[0]:pos_ref[0] + chunk]
                        if chunk < frames:
                            outdata[chunk:] = 0
                        pos_ref[0] += chunk

                    # 每次（每句）播放前重新读取设备配置，实现热更新
                    _dev = self.device_id
                    try:
                        _cfg_id = get_config("audio.device_id", None)
                        _dev = int(_cfg_id) if _cfg_id is not None else self.device_id
                    except Exception:
                        pass

                    # 尝试指定设备，若失败则回退到系统默认输出
                    _try_list = ([_dev, None] if _dev is not None else [None])
                    for _try_dev in _try_list:
                        pos_ref[0]   = 0
                        _first_cb[0] = True   # 重试时重置，重新等待真实出声
                        try:
                            with sd.OutputStream(
                                samplerate=sr, channels=channels,
                                device=_try_dev, dtype='float32',
                                callback=callback, blocksize=1024
                            ):
                                while pos_ref[0] < len(data) and not self._halted and not self._skip_current:
                                    time.sleep(0.01)
                            break  # 播放成功，退出重试循环
                        except Exception as _e:
                            if _try_dev is None:
                                log.error("[AudioPlayer] 播放失败 (系统默认设备): %s", _e)
                            else:
                                log.warning("[AudioPlayer] 设备 %s 播放失败: %s，回退到系统默认", _try_dev, _e)

                    if self._skip_current:
                        self._skip_current = False
                        log.info("[AudioPlayer] 播放被打断")

                except Exception as e:
                    log.error("[AudioPlayer] 音频解码失败: %s", e)

                # 等待 0.2 秒缓冲一下，防止 VAD 在播放刚结束的一瞬间触发
                time.sleep(0.2)
                self.set_finished_async()
                self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        '''停止播放器线程'''
        self._stop_flag = True
        self.queue.put(None)

    def clear_and_stop(self):
        '''立即停止当前播放、清空队列、阻止后续一切播放'''
        self._halted = True
        # 清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break
        # 停止当前播放
        self._skip_current = True

    def resume(self):
        '''恢复播放能力（下一轮对话开始时调用）'''
        self._halted = False
        self.has_started_playing = False  # 新一轮对话，重置出声标记

class MonikaBus:
    def __init__(self, vts_manager: VTSManager, tts: BaseTTS = None):
        from core.tts_client import GPTSoVITSTTS
        self.vts = vts_manager
        self.tts: BaseTTS = tts if tts is not None else GPTSoVITSTTS()
        
        # GUI 初始化后由外部注入，None 时静默跳过
        self.live2d = None
        
        # 口型同步回调（供 UI debug 面板订阅），签名: callback(open_y: float, scores: np.ndarray)
        self.lipsync_callback = None
        
        # 逐句显示回调（供 UI 聊天框订阅），签名: callback(sentence: str, is_first: bool)
        self.sentence_callback = None
        
        # TTS 合成完成回调（供 Brain 背压控制），签名: callback()
        self.tts_done_callback = None
        self._interrupted = False
        
        try:
            self.device_id = self.set_device_id()
        except Exception as e:
            log.warning("[Bus] 获取音频设备 ID 失败，没有装VB兄弟: %s", e)
            self.device_id = None
        
        self.loop = asyncio.get_running_loop()
        self.player = AudioPlayer(self.device_id, self.loop)
        self.player.start()

        self.text_queue = asyncio.Queue(maxsize=10)
        self.audio_queue = asyncio.Queue(maxsize=10)

        # 记录上次对话完成的时间戳
        self._last_completion_time = 0

        asyncio.create_task(self._text_worker())# TTS 合成 worker
        asyncio.create_task(self._audio_player_worker())# 音频播放 worker
        asyncio.create_task(self._tts_health_check_loop())
        
        self.action_pattern  = re.compile(r"\[(.*?)\]")
        self.emotion_pattern = re.compile(r"\((.*?)\)")
        
        # 单个句尾标点断句；排除连续多点（...）和汉字省略号（……）
        self.sentence_split  = re.compile(r"([！？!?\n]|…+|(?<![.\u2026])\.(?![.\u2026]))")
        self.garbage_pattern = re.compile(r"^[！？!.?\u2026…\s\n]+$")
        # self._code_block_re = re.compile(r'```[\s\S]*?```')
        # self._chinese_meta_re = re.compile(r'[（][^）]*[）]|[【][^】]*[】]')

    @staticmethod
    def _strip_code_for_tts(text: str) -> str:
        '''移除 Markdown 代码块（```...```），替换为简短占位符'''
        return re.sub(r'```[\s\S]*?```', '[代码块]', text)

    @staticmethod
    def _strip_chinese_meta_for_tts(text: str) -> str:
        '''移除中文全角括号（）和方头括号【】及其内容（防 LLM 乱造表情）'''
        return re.sub(r'[（][^）]*[）]|[【][^】]*[】]', '', text)

    def _l2d(self, method: str, *args):
        '''非阻塞地在 Live2DWidget 上调用指定方法（未注入时静默跳过）'''
        if self.live2d is None:
            return
        fn = getattr(self.live2d, method, None)
        if fn:
            fn(*args)

    @property
    def is_interrupted(self):
        return self._interrupted

    def reset_interrupt(self):
        '''每轮对话开始前调用，重置打断状态并恢复播放能力'''
        self._interrupted = False
        self.player.resume()

    def set_device_id(self):
        # 先检索指定设备 ID
        direct_id = get_config("audio.device_id", None)
        if direct_id is not None:
            try:
                info = sd.query_devices(int(direct_id))
                log.info("[Bus] 直接使用设备 ID=%d: %s", direct_id, info['name'])
                return int(direct_id)
            except Exception as e:
                log.warning("[Bus] audio.device_id=%s 无效: %s，回退到关键字搜索", direct_id, e)

        # 回退到按关键字搜索
        keyword = get_config("audio.device_keyword", "Voicemeeter Input")
        host_api = get_config("audio.host_api", 0)
        for idx, device in enumerate(sd.query_devices()):
            if keyword in device['name'] and device['hostapi'] == host_api:
                log.info("[Bus] Found %s at %s", keyword, device['index'])
                return device['index']
        log.warning("[Bus] 找不到音频设备: %s", keyword)
        return None

    async def _text_worker(self):
        '''文本处理：TTS 合成 -- 音频队列TTS 合成与音频播放并行，实现流水线'''
        while True:
            task = await self.text_queue.get()
            if task is None:
                self.text_queue.task_done()
                break

            if self._interrupted:
                self.text_queue.task_done()
                continue

            text, emotions, motions = task
            log.debug("[Bus-Worker] 合成: %s", text)

            bench.tick()
            wav_bytes = await self.tts.synthesize(text)
            bench.tock("tts_synth_ms")
            bench.record("sentence_count", 1)

            if wav_bytes and not self._interrupted:
                await self.audio_queue.put((wav_bytes, emotions, motions, text))
            elif not wav_bytes and not self._interrupted:
                log.warning("[Bus-Worker] TTS 返回空数据，跳过该句")

            # 背压：TTS 合成完成才通知 Brain 可以继续生成
            if self.tts_done_callback:
                try:
                    self.tts_done_callback()
                except Exception:
                    pass

            self.text_queue.task_done()

    async def _audio_player_worker(self):
        '''消费方式: 按序号保序播放音频，支持乱序到达缓冲'''
        
        async def _play_one(wav_bytes, emotions, motions, text):
            # -- 逐句回调：音频开始播放时通知 UI 显示 --
            if self.sentence_callback and text:
                try:
                    self.sentence_callback(text, False)
                except Exception:
                    pass
            for emo in emotions:
                await self.vts.set_expression(emo)
                self._l2d('set_emotion', emo)
            for mot in motions:
                await self.vts.trigger_action_exp(mot)
                self._l2d('trigger_motion', mot)

            # -- 口型同步预分析（播放前完成，播放时同步播放）------------------
            self.player.has_started_playing = False
            _lipsync_task = None
            try:
                _ls_times, _ls_oy, _ls_sc = _get_lipsync().analyze_wav_bytes(wav_bytes)
                _lipsync_task = asyncio.create_task(
                    self._drive_lipsync(_ls_times, _ls_oy, _ls_sc)
                )
            except Exception as _lse:
                log.debug("[Bus] 口型同步分析跳过: %s", _lse)

            # 同步将 event clear：必须在协程内直接调用，而不能用 call_soon_threadsafe
            # asynio.Event.wait() 在 event 已 set 时不会 yield，导致调度的 clear 回调永远跟不上
            self.player.finished_event_async.clear()
            self.player.add_wav(wav_bytes)
            bench.tick()
            await self.player.finished_event_async.wait()
            bench._current.audio_play_ms += bench.tock("_discard")

            # 播放结束后取消口型分析任务
            if _lipsync_task and not _lipsync_task.done():
                _lipsync_task.cancel()
                try:
                    await _lipsync_task
                except asyncio.CancelledError:
                    pass
            
            if motions:
                # 表情持续时间
                motion_hold_time = float(get_config("vts.motion_hold_time", 0.5))
                async def _reset_action_delayed():
                    await asyncio.sleep(motion_hold_time)
                    if not self._interrupted:
                        await self.vts.reset_action_exp()
                asyncio.create_task(_reset_action_delayed())

        while True:
            item = await self.audio_queue.get()
            if item is None:
                self.audio_queue.task_done()
                break

            if self._interrupted:
                self.audio_queue.task_done()
                continue

            wav_bytes, emotions, motions, text = item
            await _play_one(wav_bytes, emotions, motions, text)
            self.audio_queue.task_done()  # 播放完成后再 task_done

    async def process_stream_input(self, text_stream):
        '''
        处理流输入
        返回 LLM 的原始输出文本（含表情/动作标记），用于保存到 brain history
        被打断时会提前返回已生成的部分，用来记录记忆（未实现）
        '''
        buffer = ""
        raw_output = ""  # LLM 原始输出
        pending_emotions = []
        pending_motions = []

        async for chunk in text_stream:
            if self._interrupted:
                break

            raw_output += chunk
            buffer += chunk
            
            actions = self.action_pattern.findall(buffer)
            for action in actions:
                pending_motions.append(action)
                buffer = buffer.replace(f"[{action}]", "")

            emotions = self.emotion_pattern.findall(buffer)
            for emotion in emotions:
                pending_emotions.append(emotion)
                buffer = buffer.replace(f"({emotion})", "")

            parts = self.sentence_split.split(buffer)
            if len(parts) > 1:
                num_complete = (len(parts) - 1) // 2
                
                for i in range(num_complete):
                    sentence = parts[2*i] + parts[2*i+1]
                    sentence = self._strip_code_for_tts(sentence)
                    sentence = self._strip_chinese_meta_for_tts(sentence)
                    sentence = sentence.strip()
                    
                    if sentence and not self.garbage_pattern.match(sentence):
                        log.debug("[Bus-Split] -> '%s' (emo=%s, mot=%s)",
                                  sentence[:60], pending_emotions, pending_motions)
                        await self.text_queue.put((sentence, list(pending_emotions), list(pending_motions)))
                        pending_emotions = []
                        pending_motions = []
                        
                buffer = parts[-1]

        if not self._interrupted and buffer.strip() and not self.garbage_pattern.match(buffer):
            buffer = self._strip_code_for_tts(buffer)
            buffer = self._strip_chinese_meta_for_tts(buffer)
            if buffer.strip():
                await self.text_queue.put((buffer, list(pending_emotions), list(pending_motions)))

        return raw_output # 留存原始输出
            
    async def interrupt(self):
        '''立即打断一切：停止生成、终止TTS请求、清空队列、停止播放'''
        if self._interrupted:
            return
        self._interrupted = True
        log.info("[Bus] 执行打断操作...")

        # 1. 立即停止音频播放 + 阻止后续播放
        self.player.clear_and_stop()

        # 2. 取消正在进行的 TTS 请求
        await self.tts.cancel()

        # 3. 清空文本队列
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except asyncio.QueueEmpty:
                break
        # 4. 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break

        await self.vts.set_expression("normal")
        self._l2d('reset_expression')
        self._l2d('set_talking', False)
        log.info("[Bus] 打断完成")

    async def _drive_lipsync(
        self,
        times: "np.ndarray",
        open_y: "np.ndarray",
        scores: "np.ndarray",
    ):
        '''
        按帧时间戳驱动 Live2D 嘴型参数，与 AudioPlayer 线程播放时钟同步
        等待 AudioPlayer 真正出声后开始计时，以补偿入队延迟
        '''
        # 等待实际出声（最多 2s）
        _deadline = time.monotonic() + 2.0
        while not self.player.has_started_playing and not self._interrupted:
            if time.monotonic() > _deadline:
                return
            await asyncio.sleep(0.01)
        if self._interrupted:
            return

        _lipsync_mod = _get_lipsync()
        
        # -- 口型细节配置参数 --
        # timing_offset_ms : 微调用，正值=嘴巴动作延迟，负值=提前（相对于真正的出声时刻）
        _offset_s  = float(get_config("lipsync.timing_offset_ms",    0))  / 1000.0
        # update_interval_ms : 实际推送参数的最短间隔，降低 Live2D 更新帧率
        _update_s  = float(get_config("lipsync.update_interval_ms",  33)) / 1000.0
        # smoothing_alpha : EMA 系数 (0~1)，越小越平滑但越滞后
        _alpha     = float(get_config("lipsync.smoothing_alpha",    0.15))

        loop  = asyncio.get_event_loop()
        start = loop.time() + _offset_s # offset微调

        # EMA 平滑状态
        _ema: dict = {n: 0.0 for n in _lipsync_mod._MOUTH_PARAM_NAMES}
        _last_push = -999.0   # 上次实际推送给 Live2D 的时间

        for i in range(len(times)):
            if self._interrupted:
                break
            delay = float(times[i]) - (loop.time() - start)
            if delay > 0.002:
                await asyncio.sleep(delay)
            oy = float(open_y[i])
            target = _lipsync_mod.scores_to_params_multi(scores[i], oy)

            # EMA 平滑：每帧都计算，但不一定每帧推送
            for n in _lipsync_mod._MOUTH_PARAM_NAMES:
                _ema[n] = _alpha * target[n] + (1.0 - _alpha) * _ema[n]

            # 限速推送
            now = loop.time() - start
            if now - _last_push >= _update_s:
                self._l2d('set_mouth_params', dict(_ema))
                _last_push = now
                if self.lipsync_callback is not None:
                    try:
                        self.lipsync_callback(oy, scores[i])
                    except Exception:
                        pass

        # 播放结束，嘴归零
        self._l2d('set_mouth_params', _lipsync_mod.MOUTH_PARAMS_RESET)
        if self.lipsync_callback is not None:
            try:
                import numpy as _np
                self.lipsync_callback(0.0, _np.zeros(5, dtype=_np.float32))
            except Exception:
                pass

    async def _tts_health_check_loop(self):
        '''调用: 定期 ping TTS 健康接口，不可达时记录告警'''
        import aiohttp
        health_url = get_config("tts.health_url", "http://127.0.0.1:5000/health")
        interval = float(get_config("tts.health_check_interval", 30))
        while True:
            await asyncio.sleep(interval)
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status != 200:
                            log.warning("[Bus] TTS 服务健康检查异常: HTTP %s", resp.status)
                        else:
                            log.debug("[Bus] TTS 健康检查通过")
            except Exception as e:
                log.warning("[Bus] TTS 服务不可达: %s", e)

    async def shutdown(self):
        '''优雅关闭'''
        await self.text_queue.put(None)
        await self.audio_queue.put(None)
        self.player.stop()
        self.player.join()
        await self.tts.close()
    
    async def wait_for_completion(self):
        '''等待 TTS 队列和音频全部播放完毕，完成后恢复表情至 normal'''
        await self.text_queue.join()
        await self.audio_queue.join()

        await self.player.finished_event_async.wait()
        
        # 记录对话完成时间，用于超时表情恢复
        self._last_completion_time = time.time()
        
        # 一轮对话完成，自动恢复表情和动作到中立状态
        log.info("[Bus] 对话完成，恢复表情至 normal")
        await self.vts.set_expression("normal")
        await self.vts.reset_action_exp()
        self._l2d('reset_expression')

    async def check_and_restore_expression_if_timeout(self, timeout_sec=2.0):
        '''检查是否超时，超时则恢复表情到 idle
        timeout_sec: 超时时间默认 2.0 秒
        '''
        if self._last_completion_time == 0:
            # 还没有任何完成过的对话
            return
        
        elapsed = time.time() - self._last_completion_time
        if elapsed > timeout_sec:
            # 未连接 VTS 时跳过
            if not self.vts.is_connected:
                return
            # 检查当前表情是否已经是 idle
            if self.vts.current_expression_id != self.vts.name_to_id.get("idle"):
                print(f">>> [Bus] 对话已闲置 {elapsed:.1f}s，恢复表情至 idle")
                await self.vts.set_expression("idle")
                # 更新时间戳，避免重复恢复
                self._last_completion_time = time.time()

    # -- Tool Calling的一些死配置 ----------------------------------

    async def on_tool_call(self, tool_name: str, params: dict):
        '''工具调用开始 -> 触发 Monika 的表情反馈'''
        _emotion_map = {
            "read_file": "curious",
            "write_file": "aBitSerious",
            "run_command": "normal",
            "list_directory": "curious",
        }
        emotion = _emotion_map.get(tool_name, "normal")
        if self.vts and self.vts.is_connected:
            await self.vts.set_expression(emotion)
        log.debug("[Bus] Tool call: %s(%s) -> emotion=%s", tool_name, params, emotion)

    async def on_tool_result(self, success: bool):
        '''工具执行完成'''
        if not success and self.vts and self.vts.is_connected:
            await self.vts.set_expression("sad")
            # 失败了就发送话语
            await self.text_queue.put(("工具执行失败了呢，我再试试", [], []))
            