
import os
import json
import threading
import numpy as np
from datetime import datetime


class SpeakerIdentifier:
    '''声纹识别类'''

    def __init__(self, data_path: str = None, threshold: float = 0.75, samplerate: int = 16000):
        '''
        Args:
            data_path:  speakers.json 的绝对/相对路径
            threshold:  相似度阈值（0~1），默认 0.75
            samplerate: 输入音频采样率（必须与 Ears 一致！默认 16000）
        '''
        self.threshold = threshold
        self.samplerate = samplerate
        self.data_path = os.path.abspath(data_path or os.path.join("memory", "data", "speakers.json"))
        self._lock = threading.Lock()
        self._encoder = None          # 在下面再初始化

        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self.registry = self._load_registry()

        # 加载编码器，懒加载会卡
        self._init_encoder()

        count = len(self.registry["speakers"])
        print(f">>> [SpeakerID] 声纹识别就绪，已注册说话人: {count} 位")
        print(f">>> [SpeakerID] 注册表路径: {self.data_path}")

    def _init_encoder(self):
        '''加载 VoiceEncoder'''
        if self._encoder is not None:
            return
        try:
            from resemblyzer import VoiceEncoder
            print(">>> [SpeakerID] 正在加载声纹编码器...")
            self._encoder = VoiceEncoder()
            print(">>> [SpeakerID] 声纹编码器加载完成 (GE2E 模型已就绪)")
        except ImportError:
            raise ImportError(
                "声纹识别要 resemblyzer 库，没有的话运行: pip install resemblyzer"
            )
        except Exception as e:
            print(f"!!! [SpeakerID] 编码器加载失败: {e}")
            raise


    def _get_encoder(self):
        '''模型用，返回已加载的编码器'''
        return self._encoder


    def _load_registry(self) -> dict:
        '''
        加载声纹注册表，用户可以直接编辑 speakers.json 文件中的 'name' 字段来改名
        '''
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 旧格式
                if "speakers" not in data:
                    data = {"speakers": [], "next_id": 1}
                return data
            except Exception as e:
                print(f"!!! [SpeakerID] 注册表读取失败，重置: {e}")
        return {"speakers": [], "next_id": 1}

    def _save_registry(self):
        try:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"!!! [SpeakerID] 注册表保存失败: {e}")

    def _reload_speaker_from_file(self, speaker_id: str) -> dict | None:
        '''
        重新加载json
        
        Returns:
            最新的说话人数据 或 None（未找到）
        '''
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                for speaker in file_data.get("speakers", []):
                    if speaker["id"] == speaker_id:
                        return speaker
        except Exception as e:
            print(f"!!! [SpeakerID] 重新加载说话人失败: {e}")
        return None

    # -- 核心算法 --------------------------------------------

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        '''余弦相似度法'''
        a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        '''
        用resemblyzer提取256维embedding，
        audio_data采样率须与 self.samplerate 一致
        格式: (frames,) 一维的数组
        '''
        from resemblyzer import preprocess_wav
        encoder = self._get_encoder()
        
        # 确保维度
        if audio_data.ndim > 1:
            audio_data = np.squeeze(audio_data)
        
        # preprocess_wav作用: 重采样、静音切除、幅值归一化
        wav = preprocess_wav(audio_data, source_sr=self.samplerate)
        return encoder.embed_utterance(wav)



    # -- 公开调用区 --------------------------------------------

    def identify(self, audio_data: np.ndarray) -> dict | None:
        '''
        识别说话人

        Args:
            audio_data: float32 mono ndarray，注意采样率

        Returns:
            {
                "id":         "speaker_001",
                "name":       "陌生人1",        
                "is_new":     False,           
                "confidence": 0.82
            }
            或 None
        '''
        
        # 过滤过短音频
        if audio_data is None or len(audio_data) < int(self.samplerate * 0.8):
            return None

        try:
            embedding = self._extract_embedding(audio_data)
        except Exception as e:
            print(f"!!! [SpeakerID] 声纹提取失败: {e}")
            return None

        with self._lock: # 这里是线程安全的
            best_speaker = None
            best_score = -1.0

            for speaker in self.registry["speakers"]:
                for stored_emb in speaker["embeddings"]:
                    score = self._cosine_similarity(embedding, stored_emb)
                    if score > best_score:
                        best_score = score
                        best_speaker = speaker

            if best_speaker and best_score >= self.threshold: # 更新声纹样本（最多 10 条）
                # TODO 可以考虑改成比较后修改最差的样本。或者加权？
                best_speaker["last_seen"] = datetime.now().isoformat()
                
                if len(best_speaker["embeddings"]) < 10:
                    best_speaker["embeddings"].append(embedding.tolist())
                self._save_registry()
                
                updated_speaker = self._reload_speaker_from_file(best_speaker["id"])
                if updated_speaker:
                    best_speaker = updated_speaker
                
                return {
                    "id":         best_speaker["id"],
                    "name":       best_speaker["name"],
                    "is_new":     False,
                    "confidence": round(best_score, 3),
                }
            else:
                #新说话人
                new_sp = self._create_new_speaker(embedding)
                self._save_registry()
                print(f">>> [SpeakerID] 新说话人已注册: {new_sp['name']}  (id={new_sp['id']})")
                print(f"    提示：可直接编辑 {self.data_path} 中的 'name' 字段改名")
                return {
                    "id":         new_sp["id"],
                    "name":       new_sp["name"],
                    "is_new":     True,
                    "confidence": 0.0,
                }

    def _create_new_speaker(self, embedding: np.ndarray) -> dict:
        '''新说话人条目，自动命名'''
        sid = f"speaker_{self.registry['next_id']:03d}"
        name = f"陌生人{self.registry['next_id']}"
        self.registry["next_id"] += 1

        speaker = {
            "id":         sid,
            "name":       name,
            "embeddings": [embedding.tolist()],
            "first_seen": datetime.now().isoformat(),
            "last_seen":  datetime.now().isoformat(),
            "notes":      "", # 备注
        }
        self.registry["speakers"].append(speaker)
        return speaker

    # def rename_speaker(self, speaker_id: str, new_name: str) -> bool:
    #     '''重命名说话人函数版本'''
    #     with self._lock:
    #         for speaker in self.registry["speakers"]:
    #             if speaker["id"] == speaker_id:
    #                 old = speaker["name"]
    #                 speaker["name"] = new_name
    #                 self._save_registry()
    #                 print(f">>> [SpeakerID] 重命名: {old} -> {new_name}")
    #                 return True
    #     print(f"!!! [SpeakerID] 未找到 id={speaker_id}")
    #     return False

    # def delete_speaker(self, speaker_id: str) -> bool:
    #     '''从注册表删除说话人，全删'''
    #     with self._lock:
    #         before = len(self.registry["speakers"])
    #         self.registry["speakers"] = [
    #             s for s in self.registry["speakers"] if s["id"] != speaker_id
    #         ]
    #         if len(self.registry["speakers"]) < before:
    #             self._save_registry()
    #             print(f">>> [SpeakerID] 已删除: {speaker_id}")
    #             return True
    #     print(f"!!! [SpeakerID] 未找到 id={speaker_id}")
    #     return False

    def get_all_speakers(self) -> list:
        '''返回所有已注册说话人的 Dict'''
        return [
            {
                "id":         s["id"],
                "name":       s["name"],
                "first_seen": s["first_seen"],
                "last_seen":  s["last_seen"],
                "samples":    len(s["embeddings"]),
                "notes":      s.get("notes", ""),
            }
            for s in self.registry["speakers"]
        ]
    
    def shutdown(self):
        '''
        优雅关闭：
        1. 最后一次保存注册表
        2. 卸载编码器
        '''
        print(">>> [SpeakerID] 正在关闭声纹识别系统...")
        
        # 最后一次保存
        with self._lock:
            self._save_registry()
        
        # 卸载编码器
        self._encoder = None
        
        print(">>> [SpeakerID] 声纹识别系统已正常关闭")
