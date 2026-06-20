# -*- coding: utf-8 -*-
"""
Shazam 音频指纹识别 - 内存数据库实现

将指纹库（music 表 + finger_prints 表）以纯 Python 数据结构存储在内存中，
替代原来的 MySQL 依赖，无需任何外部数据库服务。

适配了 IConnector 抽象接口，调用方无需修改业务逻辑。

数据结构:
    _music_table:  List[dict]  {music_id, music_name, music_path}
    _finger_prints: Dict[int, List[Tuple[str, int]]]   music_id -> [(hash, offset), ...]
    _hash_index:    Dict[str, List[Tuple[int, int]]]   hash -> [(music_id, offset), ...]

线程安全: 使用 threading.RLock
"""

import threading
from typing import List, Dict, Tuple, Set, Optional

from ..utils.print_utils import print_message, print_error


class _MemDB:
    """
    进程级共享的内存数据库实例。
    所有 InMemoryConnector 实例共享同一个 _MemDB，
    保证指纹库在多线程/多组件下的一致性。
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._lock = threading.RLock()
        self._music_table: List[dict] = []
        self._finger_prints: Dict[int, List[Tuple[str, int]]] = {}
        self._hash_index: Dict[str, List[Tuple[int, int]]] = {}
        self._next_music_id = 1

    # ---------- 表操作 ----------

    def find_music_by_path(self, music_path: str) -> Optional[int]:
        with self._lock:
            for m in self._music_table:
                if m["music_path"] == music_path:
                    return m["music_id"]
        return None

    def find_music_by_name(self, music_name: str) -> Optional[int]:
        with self._lock:
            for m in self._music_table:
                if m["music_name"] == music_name:
                    return m["music_id"]
            for m in self._music_table:
                if music_name in m["music_name"]:
                    return m["music_id"]
        return None

    def find_music_name_by_id(self, music_id: int) -> Optional[str]:
        with self._lock:
            for m in self._music_table:
                if m["music_id"] == music_id:
                    return m["music_name"]
        return None

    def find_music_path_by_id(self, music_id: int) -> Optional[str]:
        with self._lock:
            for m in self._music_table:
                if m["music_id"] == music_id:
                    return m["music_path"]
        return None

    def add_music(self, music_name: str, music_path: str) -> int:
        with self._lock:
            music_id = self._next_music_id
            self._next_music_id += 1
            self._music_table.append({
                "music_id": music_id,
                "music_name": music_name,
                "music_path": music_path,
            })
        self.auto_save()
        return music_id

    def update_music_info(self, music_id: int, music_name: str = None, music_path: str = None):
        """更新音乐的名称和/或路径"""
        result = False
        with self._lock:
            for m in self._music_table:
                if m["music_id"] == music_id:
                    if music_name is not None:
                        m["music_name"] = music_name
                    if music_path is not None:
                        m["music_path"] = music_path
                    result = True
                    break
        if result:
            self.auto_save()
        return result

    def store_finger_prints(self, music_id: int, hashes: List[Tuple[str, int]]):
        with self._lock:
            self._finger_prints[music_id] = []
            for h, offset in hashes:
                self._finger_prints[music_id].append((h, offset))
                self._hash_index.setdefault(h, []).append((music_id, offset))
        self.auto_save()

    def count_hash_by_music(self, music_id: int) -> int:
        with self._lock:
            return len(self._finger_prints.get(music_id, []))

    def find_matching_hashes(self, query_hashes: List[Tuple[str, int]]):
        """
        生成与 query_hashes 中哈希匹配的记录。

        Yields:
            (music_id, db_offset, query_offset)
        """
        mapper = {h: offset for h, offset in query_hashes}
        with self._lock:
            for h, query_offset in query_hashes:
                entries = self._hash_index.get(h)
                if entries is None:
                    continue
                for music_id, db_offset in entries:
                    yield music_id, db_offset, mapper[h]

    def get_fingerprints_by_music_id(self, music_id: int) -> List[Tuple[str, int]]:
        """获取指定音乐的所有指纹"""
        with self._lock:
            return list(self._finger_prints.get(music_id, []))

    def get_all_music(self) -> List[Tuple[int, str]]:
        """获取所有音乐"""
        with self._lock:
            return [(m["music_id"], m["music_name"]) for m in self._music_table]

    def get_all_references(self):
        with self._lock:
            refs = []
            for m in self._music_table:
                refs.append({
                    "music_id": m["music_id"],
                    "name": m["music_name"],
                    "path": m["music_path"],
                    "hash_count": len(self._finger_prints.get(m["music_id"], [])),
                })
        return refs

    def delete_reference(self, music_id: int) -> bool:
        with self._lock:
            removed_entries = self._finger_prints.pop(music_id, None)
            if removed_entries is not None:
                for h, _ in removed_entries:
                    entries = self._hash_index.get(h)
                    if entries:
                        entries[:] = [(mid, off) for mid, off in entries if mid != music_id]
                        if not entries:
                            self._hash_index.pop(h, None)
            self._music_table = [m for m in self._music_table if m["music_id"] != music_id]
        self.auto_save()
        return True

    def clear(self):
        """清空全部数据（慎用，不会自动保存到磁盘）"""
        with self._lock:
            self._music_table.clear()
            self._finger_prints.clear()
            self._hash_index.clear()
            self._next_music_id = 1

    def stats(self) -> dict:
        with self._lock:
            total = sum(len(v) for v in self._finger_prints.values())
            return {
                "music_count": len(self._music_table),
                "total_hashes": total,
                "unique_hashes": len(self._hash_index),
            }

    def export(self) -> dict:
        """导出全部内存数据，可用于序列化到磁盘。"""
        with self._lock:
            return {
                "music_table": [dict(m) for m in self._music_table],
                "finger_prints": {
                    mid: list(v) for mid, v in self._finger_prints.items()
                },
                "hash_index": {
                    h: list(v) for h, v in self._hash_index.items()
                },
                "next_music_id": self._next_music_id,
            }

    def load(self, data: dict):
        """从 export() 的结果恢复。"""
        with self._lock:
            self._music_table = [dict(m) for m in data.get("music_table", [])]
            self._finger_prints = {
                int(k): [tuple(vv) for vv in v]
                for k, v in data.get("finger_prints", {}).items()
            }
            self._hash_index = {
                h: [tuple(vv) for vv in v]
                for h, v in data.get("hash_index", {}).items()
            }
            self._next_music_id = int(data.get("next_music_id", 1))

    # ---------- 磁盘持久化 ----------

    _storage_path: str = None
    _auto_save_enabled: bool = True

    def set_storage_path(self, path: str):
        """设置持久化文件路径（启用后会自动保存/加载）"""
        self._storage_path = path

    def get_storage_path(self) -> Optional[str]:
        return self._storage_path

    def save_to_disk(self, path: str = None):
        """将内存数据序列化到磁盘 JSON 文件"""
        target = path or self._storage_path
        if not target:
            return
        import json, os
        data = self.export()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self, path: str = None) -> bool:
        """从磁盘 JSON 文件恢复内存数据，返回是否成功加载"""
        target = path or self._storage_path
        if not target:
            return False
        import json, os
        if not os.path.exists(target):
            return False
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load(data)
            return True
        except Exception:
            return False

    def auto_save(self):
        """如果设置了路径且启用自动保存，则写入磁盘"""
        if self._storage_path and self._auto_save_enabled:
            self.save_to_disk()


class InMemoryConnector:
    """
    兼容 IConnector 抽象接口的内存数据库连接器。

    使用方式：
        connector = InMemoryConnector()
        # 像 MySQLConnector 一样使用
    """

    def __init__(self):
        self.db = _MemDB()

    # ---------- 兼容 MySQLConnector 的属性 ----------
    # 注意：stft_create / stft_predict 的签名都用的是 `connector` 对象，
    # 调用这些属性/方法在内存实现里并不存在；在 API 层不再依赖 cursor/conn。
    # 这里仍然提供，以兼容老代码对 `cursor.close()` / `conn.close()` 的潜在访问。
    @property
    def cursor(self):
        return self

    @property
    def conn(self):
        return self

    def close(self):
        return None

    def commit(self):
        return None

    # ---------- 实现 IConnector 的所有抽象方法 ----------

    def store_finger_prints(self, hashes, music_id_fk):
        self.db.store_finger_prints(int(music_id_fk), [(str(h), int(o)) for h, o in hashes])

    def _add_finger_print(self, item, music_id_fk):
        self.store_finger_prints([item], int(music_id_fk))

    def find_music_by_music_path(self, music_path):
        return self.db.find_music_by_path(str(music_path))

    def find_music_name_by_music_id(self, music_id):
        return self.db.find_music_name_by_id(int(music_id))

    def find_music_path_by_music_id(self, music_id):
        return self.db.find_music_path_by_id(int(music_id))

    def find_music_by_music_name(self, music_name):
        return self.db.find_music_by_name(str(music_name))

    def calculation_hash_num_by_music_id(self, music_id):
        return self.db.count_hash_by_music(int(music_id))

    def add_music(self, music_path, music_name=None):
        path_str = str(music_path)
        if music_name is None:
            import os.path
            music_name = os.path.splitext(os.path.basename(path_str))[0]
        return self.db.add_music(str(music_name), path_str)

    def update_music_info(self, music_id, music_name=None, music_path=None):
        return self.db.update_music_info(int(music_id), music_name, music_path)

    def _find_finger_print(self, hash_):
        entries = self.db._hash_index.get(hash_)
        if not entries:
            return None
        for mid, offset in entries:
            return (mid, offset)
        return None

    def find_math_hash(self, hashes):
        for mid, db_offset, q_offset in self.db.find_matching_hashes(list(hashes)):
            yield mid, db_offset, q_offset

    def find_math_hash_old(self, hashes):
        yield from self.find_math_hash(hashes)

    def get_fingerprints_by_music_id(self, music_id):
        return self.db.get_fingerprints_by_music_id(int(music_id))

    def get_all_music(self):
        return self.db.get_all_music()


class InMemoryDatabaseChecker:
    """
    对应原来的 DatabaseChecker，这里不需要做任何真正的连接/建表。
    启动时自动从磁盘恢复数据，注册退出时自动保存。
    """

    # 默认持久化文件路径（相对于项目根目录）
    DEFAULT_STORAGE_FILENAME = "data/shazam_db.json"

    def check_database(self):
        db = _MemDB()

        # 确定存储路径
        if not db.get_storage_path():
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # in_memory.py → database/ → shazam/ → core/ → backend/ → 项目根目录（4层）
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
            storage_path = os.path.join(project_root, self.DEFAULT_STORAGE_FILENAME)
            db.set_storage_path(storage_path)

        # 尝试从磁盘恢复
        if db.load_from_disk():
            stats = db.stats()
            print_message(
                f"Shazam 内存数据库已从磁盘恢复: "
                f"{stats['music_count']} 首曲目, "
                f"{stats['total_hashes']} 个指纹, "
                f"{stats['unique_hashes']} 个唯一哈希"
            )
        else:
            print_message("Shazam 内存数据库已就绪（无外部依赖，首次启动无历史数据）")

        return True

    def check_tables(self):
        return True

    @staticmethod
    def save_on_exit():
        """进程退出时自动保存（由 main.py lifespan 注册，静默执行）"""
        db = _MemDB()
        path = db.get_storage_path()
        if path:
            db.save_to_disk()


def reset_in_memory_db():
    """重置内存数据库（主要用于测试）。"""
    _MemDB().clear()
