# Shazam 算法数据库变更 —— 迁移到进程内内存数据库

- 时间: 2026-06-20 14:58:36 +0800
- 负责人: 自动完成
- 状态: 已完成

## 背景

Shazam 音频指纹识别模块原先强依赖 MySQL（`pymysql` + `fingerprint` 数据库，两张表 `music` / `finger_prints`）。
部署到无 MySQL 的环境（例如仅具备 Python + CUDA 的离线/边缘服务器）时会直接失败。

本次改造把指纹库迁移到**进程内内存**，彻底移除对任何外部数据库服务的强依赖，
同时保留切换到 MySQL 的能力（`backend="mysql"`）。

## 现状分析

核心数据流:

```
stft_create.py  创建指纹 (hash, offset)  ──►  写入数据库  ──┐
                                                              ├──►  stft_predict.py  读库匹配 (hash) ──►  时间对齐投票
                                                              │
API 层 (AudioFingerprinter / ParallelAudioFingerprinter)  ◄──┘
```

原先的依赖:

- `database/connector.py` 中的 `MySQLConnector`：实现了 `IConnector` 抽象接口
- 所有方法 (`find_music_by_music_path`, `store_finger_prints`, `find_math_hash` 等) 都是 SQL 字符串

## 改造方案

### 新增: `database/in_memory.py`

实现了两个组件:

1. `_MemDB`（进程级单例，带 `threading.RLock`）
   - `_music_table`: `List[dict]` — 等价于 `music` 表
   - `_finger_prints`: `Dict[int, List[(hash, offset)]]` — music_id -> 指纹列表
   - `_hash_index`: `Dict[str, List[(music_id, offset)]]` — 倒排索引，等价于 `finger_prints` 表上的索引
   - `export()` / `load()`: 支持序列化/反序列化（为后续持久化到磁盘做准备）
   - 所有方法都加了锁，线程安全

2. `InMemoryConnector`
   - 完全实现 `IConnector` 抽象接口
   - 同时提供 `cursor` / `conn` / `close()` / `commit()` 兼容 `MySQLConnector` 使用方

3. `InMemoryDatabaseChecker`
   - 对应原来的 `DatabaseChecker`，内存模式下只是空操作占位

### 修改: `api.py`

- `AudioFingerprinter.__init__` 新增 `backend` 参数，默认 `"memory"`
- 默认使用 `InMemoryConnector`
- `get_all_references` / `delete_reference` / `clear_database` 针对 InMemory 分支走内存 API，不再拼 SQL

### 修改: `parallel.py`

- `ThreadLocalFingerprinter` / `ParallelAudioFingerprinter` 新增 `backend` 参数
- 多线程下每个线程持有独立 `AudioFingerprinter`，共享同一个进程内 `_MemDB` 单例

### 修改: `database/__init__.py`

- 同时导出 `InMemoryConnector` / `InMemoryDatabaseChecker`

### 兼容性

- 老代码显式指定 `backend="mysql"` 仍可使用 MySQL
- 不指定 backend 的老代码默认走内存（迁移友好）
- 外部调用方（`start_server.py`, `dataset_builder*.py`, `local_monitor_service.py`, `client_monitor_service.py`, `reference_audio.py`, `precise_segment_locator` 等）**无需任何修改**

## 验证结果

```
default backend connector: InMemoryConnector
init_database OK
refs: []
ALL OK
```

## 第二期完善（2026-06-20 15:23:36 +0800）

第一期完成后，核心 `AudioFingerprinter` 已默认走内存，但以下模块仍直接创建 `MySQLConnector()` 并使用原始 SQL，
在无 MySQL 环境下会失败。本轮修复了所有这些模块。

### 修改清单

| 文件 | 修改内容 |
|------|---------|
| `database/connector.py` | `IConnector` 新增 `get_fingerprints_by_music_id` / `get_all_music` 抽象方法；`MySQLConnector` 实现 |
| `database/in_memory.py` | `_MemDB` 新增 `get_fingerprints_by_music_id` / `get_all_music`；`InMemoryConnector` 委托方法 |
| `parallel.py` | 删除无用 `MySQLConnector` import；`ThreadLocalFingerprinter` / `ParallelAudioFingerprinter` / 便捷函数全部补齐 `backend` 参数 |
| `matching_engine.py` | `build_index_from_database()` 原始 SQL → `get_all_music()` + `get_fingerprints_by_music_id()` |
| `precise_segment_locator/locator.py` | Import/默认连接器 改为 `InMemoryConnector`；3 处原始 SQL 替换为接口方法 |
| `client_monitor_service.py` | `_init_analyzer` 中 `MySQLConnector()` → `InMemoryConnector()`，原始 SQL → `get_all_music()` |
| `local_monitor_service.py` | `_init_analyzer` + `_update_reference_audios` 两处同上 |
| `create_database.py` | 修复 import 路径 `.database.MySQLConnector` → `.database.connector` |

### 验证结果

```
InMemoryConnector OK
get_all_music: []
add_music: 1
get_fingerprints_by_music_id: []
after store: [('abc123', 0), ('def456', 10)]
hash_count: 2
find_math_hash: [(1, 0, 0)]
get_all_music: [(1, 'test')]

ThreadLocalFingerprinter backend=memory OK
ParallelAudioFingerprinter backend=memory OK
backend attr: memory
thread_local backend: memory
```

### 仍使用 MySQLConnector 的文件（可后续处理）

- `backend/api/dataset_builder.py` — 3 处，仅用作 `LongAudioAnalyzer` 参数，可简单切换
- `backend/api/dataset_builder_v2.py` — 1 处，同上
- `backend/core/shazam/create_database.py` — `add_music_fp_to_database()` 工具函数
- `backend/core/precise_segment_locator/example.py` — 示例文件
- `backend/core/long_audio_analyzer/examples.py` — 示例文件

## 第三期：reference_audio.py API 适配 + 持久化（2026-06-20）

### 问题
- `reference_audio.py` 4 个接口直接使用原始 SQL，调用 `InMemoryConnector.cursor.execute()` 导致 500 错误
- 内存数据库重启后数据全部丢失

### 修改

#### reference_audio.py — 4 处原始 SQL 替换为接口方法
| 接口 | 原 SQL | 改为 |
|------|--------|------|
| `GET /list` | `SELECT ... LEFT JOIN ...` | `fp.get_all_references()` |
| `POST /upload` | `UPDATE music SET ...` | `connector.update_music_info()` |
| `POST /add-existing` | `INSERT INTO music ...` | `connector.add_music()` |
| `DELETE /{id}` | `SELECT music_name FROM music` | `connector.find_music_name_by_music_id()` |

#### 基础设施新增
- `_MemDB.update_music_info(music_id, name, path)` — 更新曲目信息
- `InMemoryConnector.update_music_info()` — 委托方法
- `MySQLConnector.update_music_info()` — SQL UPDATE 实现

#### 磁盘持久化
- **存储文件**: `<项目根>/data/shazam_db.json`（JSON 格式）
- **自动保存**: `add_music` / `store_finger_prints` / `update_music_info` / `delete_reference` 操作后立即写入磁盘
- **自动恢复**: `InMemoryDatabaseChecker.check_database()` 启动时从磁盘加载
- **退出保护**: `atexit` 注册，进程退出时自动保存
- `_MemDB.save_to_disk(path)` / `load_from_disk(path)` — 手动控制接口
- `clear()` 不会触发自动保存（保护磁盘数据不被误清）

#### 启动脚本
- `start_server.py` 第 5 步：`InMemoryDatabaseChecker.check_database()` 自动处理加载+打印状态
- `start_server.sh` `check_database()`：Python 子进程初始化，输出曲目数/指纹数

### 验证结果
```
# 持久化
写入后: 曲目=1, 指纹=3
重置后: 曲目=0
恢复后: 曲目=1, 指纹=3  ← 数据完整恢复
find_music_by_path: OK
find_music_by_name: OK
update_music_info 后: name=新名称, path=/final/path.wav ← 更新也持久化

# InMemoryConnector 完整流程
写入后: {'music_count': 1, 'total_hashes': 3, 'unique_hashes': 2}
恢复后: {'music_count': 1, 'total_hashes': 3, 'unique_hashes': 2}
```

## 后续可选项

1. ~~持久化~~ ✅ 已实现：自动保存到 `data/shazam_db.json`，启动时自动恢复
2. 分布式：多进程/多机可把 `_MemDB` 换成 Redis/LevelDB 后端，保持 `InMemoryConnector` API 不变即可
3. 大数据量优化：当前每次变更都全量写入 JSON，指纹库超过 10 万条时可改为增量写入或使用 SQLite 文件
