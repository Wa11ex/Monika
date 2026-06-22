import os
import json
import chromadb
from datetime import datetime, timezone
from typing import List, Dict, Optional
import uuid
from utils.config_loader import get_config
from utils.logger import get_logger
import threading

from memory.memory_weights import MemoryWeights

log = get_logger(__name__)


class MemoryManager:
    '''
    记忆管理器
    
    功能:
    1. 短期记忆: 当前会话的对话历史 (在Brain.history，session文件中)
    2. 长期记忆: 所有历史对话的向量检索 (ChromaDB)
    3. 用户画像: 用户偏好、角色设定 (chr.json)
    '''
    
    def __init__(self, memory_dir="./memory"):
        
        if not isinstance(memory_dir, str):
            memory_dir = "./memory"
        
        self.memory_dir = os.path.abspath(memory_dir)
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # 数据路径加载
        db_path_from_config = get_config("memory.vector_db.db_path", os.path.join(self.memory_dir, "data", "chroma_db"))
        if not isinstance(db_path_from_config, str):
            db_path_from_config = os.path.join(self.memory_dir, "data", "chroma_db")
        self.db_path = db_path_from_config
        
        profile_path_from_config = get_config("memory.profile.path", os.path.join(self.memory_dir, "chr.json"))
        if not isinstance(profile_path_from_config, str):
            profile_path_from_config = os.path.join(self.memory_dir, "chr.json")
        self.profile_path = profile_path_from_config
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 初始化
        self._init_vector_store()
        # 用户画像
        self.profile = self._load_profile()
        
        # 短期记忆
        self.current_session = []
        self.session_start_time = datetime.now()
        
        # 追踪后台线程
        self._background_threads = []
        
        # 记忆权重（重要度评分 + 时间衰减）
        self.weights = MemoryWeights()
        self._weights_enabled = get_config("memory.weights.enable_decay", True)
        
        log.info("记忆系统初始化完成")
        log.info("数据库: %s", self.db_path)
        log.info("已存储对话: %d 条", self.collection.count())
        log.info("Lore 集合: %d 条", self.lore_collection.count())
    
    def _init_vector_store(self):
        '''初始化ChromaDB向量数据库，使用多语言嵌入模型以支持中文语义检索'''
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            # 强制 CPU 运行：嵌入查询不在实时推理路径上，不应占用宝贵显存
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                device="cpu"
            )
            log.info("嵌入模型: paraphrase-multilingual-MiniLM-L12-v2 (CPU 模式)")
        except Exception as e:
            log.warning("多语言嵌入模型加载失败，降级为默认模型: %s", e)
            self._embedding_fn = None

        try:
            # 持久化
            self.client = chromadb.PersistentClient(path=self.db_path)

            collection_name = get_config(
                "memory.vector_db.collection_name", "monika_conversations"
            )
            # 获取或创建 collection，注入嵌入函数
            kwargs = {"name": collection_name, "metadata": {"description": "Monika的长期对话记忆"}}
            if self._embedding_fn is not None:
                kwargs["embedding_function"] = self._embedding_fn
            self.collection = self.client.get_or_create_collection(**kwargs)

            # lore 专用 collection（角色前世记忆，按需语义召回）
            lore_kwargs = {"name": "monika_lore", "metadata": {"description": "Monika角色前世记忆"}}
            if self._embedding_fn is not None:
                lore_kwargs["embedding_function"] = self._embedding_fn
            self.lore_collection = self.client.get_or_create_collection(**lore_kwargs)
        except Exception as e:
            log.error("ChromaDB初始化失败: %s", e)
            log.error("请先安装: pip install chromadb sentence-transformers")
            raise
    
    #  Lore：角色前世记忆的种子写入 & 语义检索
    def seed_lore(self, chunks_path: str):
        '''
        幂等地将 lore_chunks.json 中的角色记忆块写入 lore_collection
        使用 upsert 自动处理新增/更新，不先清除旧数据（避免误删）
        '''
        if not hasattr(self, 'lore_collection'):
            return
        if not os.path.isfile(chunks_path):
            log.info("未找到 lore 文件，跳过种子写入: %s", chunks_path)
            return
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        ids, docs, metas = [], [], []
        for c in chunks:
            ids.append(c["id"])
            docs.append(c["text"])
            metas.append({"keywords": ",".join(c.get("keywords", []))})

        if ids:
            self.lore_collection.upsert(ids=ids, documents=docs, metadatas=metas)
            log.info("Lore 种子写入完成，共 %d 块", len(ids))

    def retrieve_lore(self, query: str, n: int = 1, threshold: float = 0.45) -> list:
        '''
        根据 query 语义检索最相关的 lore 块
        query:     用户当前输入
        n:         最多返回条数
        threshold: Chroma L2 距离threshold
        返回：
            [{"id": ..., "text": ..., "distance": ...}, ...]
        '''
        if not hasattr(self, 'lore_collection') or self.lore_collection.count() == 0:
            return []
        try:
            results = self.lore_collection.query(
                query_texts=[query],
                n_results=min(n, self.lore_collection.count()),
                include=["documents", "distances", "metadatas"]
            )
            hits = []
            ids = results.get("ids", [[]])[0]
            for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
                if dist < threshold:
                    hit_id = ids[i] if i < len(ids) else "?"
                    hits.append({"id": hit_id, "text": doc, "distance": dist})
            return hits
        except Exception as e:
            log.error("retrieve_lore 失败: %s", e)
            return []

    # ---------------------------------------------------------

    def _load_profile(self) -> Dict:
        '''加载用户画像，若不存在则创建含基础字段的默认结构'''
        if os.path.exists(self.profile_path):
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 兼容旧格式：补齐缺失字段
            changed = False
            for field, default in [("important_facts", []), ("preferences", {})]:
                if field not in data:
                    data[field] = default
                    changed = True
            if changed:
                self._save_profile(data)
            return data
        else:
            log.info("未找到用户画像，创建默认画像")
            default = {"name": "User", "important_facts": [], "preferences": {}}
            self._save_profile(default)
            return default
    
    def _save_profile(self, profile: Dict):
        '''保存用户画像'''
        profile["last_updated"] = datetime.now().isoformat()
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    
    def _get_lore_by_id(self, lore_id: str) -> dict | None:
        '''按 ID 精确获取 lore 块（用于硬规则触发）

        Returns:
            {"id": ..., "text": ..., "distance": 0.0} 或 None
        '''
        if not hasattr(self, 'lore_collection') or self.lore_collection.count() == 0:
            return None
        try:
            result = self.lore_collection.get(ids=[lore_id])
            if result and result.get("documents"):
                return {
                    "id": lore_id,
                    "text": result["documents"][0],
                    "distance": 0.0,  # 硬规则距离==0
                }
        except Exception:
            pass
        return None

    def add_conversation(self, user_msg: str, assistant_msg: str, metadata: Dict = None):
        '''
        存储一轮对话到长期记忆
            user_msg: 用户消息
            assistant_msg: 助手回复
            metadata: 额外元数据 (表情、动作等)
        '''
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # 构建文档
        document = f"user: {user_msg}\nassistant: {assistant_msg}"
        
        # 构建元数据
        meta = {
            "user_msg": user_msg,
            "assistant_msg": assistant_msg,
            "timestamp": timestamp,
            "session_id": self.session_start_time.isoformat()
        }
        if metadata:
            meta.update(metadata)
        
        # -- 记忆 weight评分 --
        try:
            importance = self.weights.score(document, metadata or {})
            meta["importance"] = importance
        except Exception as e:
            log.debug("记忆评分失败: %s", e)
            meta["importance"] = 0.5
        
        # 存入向量数据库
        try:
            self.collection.add(
                documents=[document],
                metadatas=[meta],
                ids=[conversation_id]
            )
        except Exception as e:
            log.error("存储对话失败: %s", e)
        
        # 也缓存到短期记忆（也是缓存
        self.current_session.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": timestamp
        })
    
    def retrieve_relevant(self, query: str, n: int = 3) -> List[Dict]:
        '''
        检索与当前查询最相关的历史对话
            query: 查询文本
            n: 返回的记忆数量
        '''
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n, self.collection.count())
            )
            
            memories = []
            if results['metadatas'] and len(results['metadatas'][0]) > 0:
                now = datetime.now(timezone.utc)
                for i, metadata in enumerate(results['metadatas'][0]):
                    # -- 记忆衰减 weight --
                    distance = results['distances'][0][i] if 'distances' in results else None
                    if self._weights_enabled:
                        try:
                            ts_str = metadata.get("timestamp", "")
                            if ts_str:
                                ts = datetime.fromisoformat(ts_str)
                                age_seconds = (now - ts).total_seconds()
                                importance = float(metadata.get("importance", 0.5))
                                decayed = self.weights.decay(importance, age_seconds)
                                if distance is not None:
                                    distance = distance * (1.0 - decayed * 0.5)
                        except Exception as e:
                            log.debug("记忆衰减计算失败: %s", e)
                    
                    memories.append({
                        "user_msg": metadata.get("user_msg", ""),
                        "assistant_msg": metadata.get("assistant_msg", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "distance": distance,
                        "importance": metadata.get("importance", None),
                    })
            
            return memories
        except Exception as e:
            log.error("检索记忆失败: %s", e)
            return []
    
    def get_memory_context(self, current_query: str, max_memories: int = 3, speaker_info: dict = None) -> str:
        '''
        生成记忆上下文,用于注入到LLM的system prompt
            current_query: 当前用户输入
            max_memories: 最大检索记忆数
            speaker_info: 说话人信息 {"name": ..., "is_new": bool, ...}，用于硬规则触发
        返回：
            格式化的记忆上下文字符串
        '''
        context_parts = []
        
        # 1. 用户画像
        user_name = self.profile.get("name") or self.profile.get("user_name")
        if user_name:
            context_parts.append(f"用户名称: {user_name}")

        job = self.profile.get("job")
        if job:
            context_parts.append(f"用户职业: {job}")

        if self.profile.get("important_facts"):
            facts = "\n".join(f"- {fact}" for fact in self.profile["important_facts"][:5])
            context_parts.append(f"关于用户的重要事实:\n{facts}")

        if self.profile.get("preferences"):
            prefs = "\n".join(f"- {k}: {v}" for k, v in list(self.profile["preferences"].items())[:3])
            if prefs:
                context_parts.append(f"用户偏好:\n{prefs}")
        
        # 2. 长期记忆检索
        if get_config("memory.enabled", False) and self.collection.count() > 0:
            memories = self.retrieve_relevant(current_query, n=max_memories)
            if memories:
                log.info("长期记忆命中 %d 条:", len(memories))
                for i, mem in enumerate(memories):
                    dist_str = f"dist={mem['distance']:.3f}" if mem.get('distance') is not None else ""
                    log.debug("    [%d] %s %s | 用户: %s...", i+1, mem['timestamp'][:10], dist_str, mem['user_msg'][:40].rstrip())
                memory_strs = []
                for mem in memories:
                    time_str = mem["timestamp"][:10] if mem["timestamp"] else "过去"
                    user_snippet = mem["user_msg"][:80].rstrip()
                    asst_snippet = mem["assistant_msg"][:80].rstrip()
                    memory_strs.append(
                        f"[{time_str}]\n  用户: {user_snippet}\n  Monika: {asst_snippet}"
                    )

                context_parts.append("【相关历史记忆】\n" + "\n\n".join(memory_strs))

        # 3. lore记忆检索
        lore_cfg = get_config("memory.lore", {})
        if isinstance(lore_cfg, dict) and lore_cfg.get("enabled", False):
            lore_threshold = lore_cfg.get("distance_threshold", 0.45)
            lore_max = lore_cfg.get("max_retrieval", 1)
        else:
            lore_threshold, lore_max = 0.0, 0  # 禁用时跳过
        if lore_max > 0:
            lore_hits = self.retrieve_lore(current_query, n=lore_max, threshold=lore_threshold)

            # -- 硬规则：speaker 名称含"陌生人"强制触发"陌生人敌意" --
            if speaker_info and "陌生人" in str(speaker_info.get("name", "")):
                forced = self._get_lore_by_id("陌生人敌意")
                if forced:
                    # 去重
                    existing_ids = {h["id"] for h in lore_hits}
                    if forced["id"] not in existing_ids:
                        lore_hits.insert(0, forced)  # 最高优先级，放在最前面
                        log.info("前世记忆硬规则触发: 陌生人敌意 (speaker=%s)",
                                 speaker_info.get("name", "未知"))

            if lore_hits:
                hit_ids = [h["id"] for h in lore_hits]
                hit_dists = [f"{h['distance']:.3f}" for h in lore_hits]
                log.info("前世记忆命中 %d 块 (dist<%s): %s",
                         len(lore_hits), lore_threshold,
                         ", ".join(f"{i}(d={d})" for i, d in zip(hit_ids, hit_dists)))
                context_parts.append("【前世记忆片段】\n" + "\n\n".join(h["text"] for h in lore_hits))

        # 组合上下文
        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return ""
    
    def add_conversation_background(self, user_msg: str, assistant_msg: str, metadata: dict = None):
        '''
        在后台线程中异步调用 add_conversation，避免阻塞主循环
        '''
        if not get_config("memory.enabled", False):
            return

        def _task():
            try:
                self.add_conversation(user_msg, assistant_msg, metadata)
                log.debug("记忆后台同步至 ChromaDB 完成")
            except Exception:
                pass  # 后台任务不应抛出未捕获异常

        t = threading.Thread(target=_task, daemon=True)
        self._background_threads.append(t)
        t.start()
    
    def update_profile(self, key: str, value):
        '''更新用户画像'''
        
        keys = key.split('.') # 兼容嵌套
        target = self.profile
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
        self._save_profile(self.profile)
        log.info("更新画像: %s = %s", key, value)
    
    def add_important_fact(self, fact: str):
        '''添加重要事实到用户画像'''
        if "important_facts" not in self.profile:
            self.profile["important_facts"] = []
        
        if fact not in self.profile["important_facts"]:
            self.profile["important_facts"].append(fact)
            self._save_profile(self.profile)
            log.info("记录重要事实: %s", fact)
    
    def summarize_session(self) -> str: # TODO
        '''
        总结当前会话
        (可以调用LLM生成摘要,这里先简单实现)
        '''
        if not self.current_session:
            return "本次会话无对话记录"
        
        summary = f"会话时长: {(datetime.now() - self.session_start_time).seconds // 60} 分钟\n"
        summary += f"对话轮数: {len(self.current_session)}\n"
        
        # 简单统计
        topics = set()
        for conv in self.current_session:
            # 这里应该用 NLP提取主题,先简化
            topics.update(conv['user'].split()[:3])
        
        summary += f"主要话题: {', '.join(list(topics)[:5])}"
        return summary
    
    def clear_session(self):
        '''清空当前会话缓存'''
        self.current_session = []
        self.session_start_time = datetime.now()
    
    def get_stats(self) -> Dict:
        '''获取记忆统计信息'''
        return {
            "total_conversations": self.collection.count(),
            "current_session_length": len(self.current_session),
            "important_facts_count": len(self.profile.get("important_facts", [])),
            "preferences_count": len(self.profile.get("preferences", {})),
            "profile_last_updated": self.profile.get("last_updated", "未知")
        }
    
    def shutdown(self):
        '''优雅关闭'''
        log.info("正在关闭记忆系统...")
        
        # 等待所有后台线程完成（或超时 10 秒）
        import time
        max_wait = 10
        start = time.monotonic()
        while self._background_threads and time.monotonic() - start < max_wait:
            # 只在有线程时才等待
            time.sleep(0.1)
        
        if self._background_threads:
            log.warning("有 %d 个后台线程超时未完成", len(self._background_threads))
        
        # 最后一次保存档案
        self._save_profile(self.profile)

        log.info("记忆系统已正常关闭")

