import asyncio
import pyvts
import os
from utils.config_loader import get_config
from utils.logger import get_logger

log = get_logger(__name__)


class VTSManager:
    def __init__(self):
        self.token_path = os.path.abspath(
            get_config("vts.token_path", "./vts_token.txt")
        )
        self.vts = pyvts.vts(plugin_info={
            "plugin_name": get_config("vts.plugin_name", "Monika_Core"),
            "developer": get_config("vts.developer", "YourName"),
            "authentication_token_path": self.token_path
        })
        self.name_to_id = {}

        self.current_expression_id = None 
        self.current_action_id = None  # 记录当前当作动作使用的 exp ID
        self.is_connected = False       # 由 connect()/close() 维护；未连接时所有操作静默跳过

    async def connect(self):
        try:
            await self.vts.connect()
            await self.vts.request_authenticate_token()
            if await self.vts.request_authenticate():
                await self.refresh_hotkey_map()
                self.is_connected = True
                log.info("[VTS] 已连接，映射表已更新")
                return True
            self.is_connected = False
            return False
        except Exception as e:
            self.is_connected = False
            log.error("[VTS] 连接失败: %s", e)
            raise

    async def refresh_hotkey_map(self):
        '''构建表情和动作的名字到 ID 的映射'''
        msg = {
            "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
            "requestID": "ListRequest", "messageType": "HotkeysInCurrentModelRequest"
        }
        res = await self.vts.request(msg)
        if 'data' in res and 'availableHotkeys' in res['data']:
            self.name_to_id = {h['name']: h['hotkeyID'] for h in res['data']['availableHotkeys']}
            self.id_to_type = {h['hotkeyID']: h['type'] for h in res['data']['availableHotkeys']}

    async def set_expression(self, name):
        '''
        专门用于切换表情的方法
        如果传入 None 或 idle， normal，则只关闭当前表情，回归模型默认状态
        
        fix：先关闭旧表情，再开启新表情（避免同时有多个表情）
        '''
        if not self.is_connected:
            return
        target_id = self.name_to_id.get(name)
    
        # -- 先关闭旧表情 --
        if self.current_expression_id and self.current_expression_id != target_id:
            log.debug("[VTS] 关闭旧表情 ID: %s", self.current_expression_id)
            await self.trigger(self.current_expression_id)
        
        await asyncio.sleep(0.1)
        
        # -- 再开启新表情 --
        if target_id:
            if self.current_expression_id == target_id: 
                log.debug("[VTS] 表情 '%s' 已在运行中", name)
            else:
                log.info("[VTS] 激活新表情: %s", name)
                await self.trigger(target_id)
        else:
            if name not in [None, "idle", "normal"]:
                log.warning("[VTS] 未找到表情 '%s'", name)

        self.current_expression_id = target_id


    async def trigger(self, hotkey_id):
        '''触发（最底层调用）失败时自动尝试重连一次；两次失败后标记断线'''
        if not self.is_connected:
            return
        msg = {
            "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
            "requestID": "Action", "messageType": "HotkeyTriggerRequest",
            "data": {"hotkeyID": hotkey_id}
        }
        try:
            await self.vts.request(msg)
        except Exception as e:
            log.warning("[VTS] 指令发送失败，尝试重连: %s", e)
            try:
                await self.connect()
                await self.vts.request(msg)
                log.info("[VTS] 重连成功，指令已重发")
            except Exception as e2:
                self.is_connected = False
                log.error("[VTS] 重连后仍失败，已标记断线: %s", e2)

    async def trigger_action_exp(self, name):
        '''
        fix： 将动作当作 exp3 触发（状态开关机制），
        与 trigger_motion 分离，以便未来支持真正的 motion3 动画
        '''
        if not self.is_connected:
            return
        target_id = self.name_to_id.get(name)
        
        # 1. 先关闭前一个来不及关的动作
        if self.current_action_id and self.current_action_id != target_id:
            log.debug("[VTS] 关闭残留短时动作 ID: %s", self.current_action_id)
            await self.trigger(self.current_action_id)
            await asyncio.sleep(0.1)

        # 2. 开启新动作
        if target_id:
            if self.current_action_id == target_id:
                log.debug("[VTS] 短时动作 '%s' 已在运行", name)
            else:
                log.info("[VTS] 激活短时动作 (模拟Exp): %s", name)
                await self.trigger(target_id)
                self.current_action_id = target_id
        else:
            log.warning("[VTS] 未找到短时动作 '%s'", name)

    async def reset_action_exp(self):
        '''
        通过再次触发快捷键，关闭当前当作动作使用的 exp3
        '''
        if self.current_action_id:
            log.debug("[VTS] 自动关闭短时动作（模拟Exp）")
            await self.trigger(self.current_action_id)
            self.current_action_id = None

    async def trigger_motion(self, name):
        '''真正的 motion3 动画触发（做保留分离）'''
        if not self.is_connected:
            return
        target_id = self.name_to_id.get(name)
        if target_id:
            log.info("[VTS] 执行 Motion3 动画: %s", name)
            await self.trigger(target_id)
        else:
            log.warning("[VTS] 未找到动作(Motion3) '%s'", name)

    async def reset_motion(self):
        '''原始的动作恢复逻辑（保留备用）'''
        motion_reset_name = get_config("vts.motion_reset_hotkey", "idle")
        reset_id = self.name_to_id.get(motion_reset_name)
        
        if reset_id:
            print(f">>> [VTS] 恢复动作: {motion_reset_name}")
            await self.trigger(reset_id)
        else:
            # 如果没有搜到，默认的回归动作
            for fallback_name in ["idle", "neutral", "normal", "None"]:
                if fallback_name in self.name_to_id:
                    print(f">>> [VTS] 恢复动作（fallback）: {fallback_name}")
                    await self.trigger(self.name_to_id[fallback_name])
                    return
            print(f">>> [VTS] 警告：未找到回归动作 '{motion_reset_name}'，请在 VTube Studio 中配置")

    async def close(self):
        if self.is_connected:
            if self.current_expression_id:
                await self.trigger(self.current_expression_id)  # 关闭表情
            if self.current_action_id:
                await self.trigger(self.current_action_id)       # 关闭动作
        self.is_connected = False
        self.current_expression_id = None
        self.current_action_id = None
        try:
            if self.vts.websocket is not None:
                await self.vts.close()
        except Exception:
            log.debug("[VTS] 关闭连接时异常（可能已断开）", exc_info=True)