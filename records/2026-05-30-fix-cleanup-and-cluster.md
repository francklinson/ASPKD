# 工作记录 — 2026-05-30

## 任务一：修复需求文档中的两个待办项

### [1] 修复任务管理中清理旧任务功能

**时间**: 2026-05-30 00:21

**问题根因**: `backend/api/tasks.py` 的 `cleanup_old_tasks()` 函数使用硬编码相对路径（`"uploads"`、`"output/vis"`、`"output/exports"`、`"output/slices"`），项目文件路径变更后，服务器工作目录可能不是项目根目录，导致路径解析失败。

**修复方案**: 在 cleanup 函数中动态计算项目根目录 `project_root`，将所有清理目录路径改为基于 `project_root` 的绝对路径。

**修改文件**:
- `backend/api/tasks.py` 第48-140行

---

### [2] 修复特征聚类功能可视化显示 Not Found

**时间**: 2026-05-30 00:21

**问题根因**: `backend/api/feature_cluster.py` 中 `interactive_html` 路径转换逻辑有两个问题：
1. 路径转换为 URL 时，条件判断 `'/visualize/' in rel_html_path` 始终为 False，因为实际路径包含 `output/vis/` 而非 `visualize/`
2. `result_dir` 使用相对路径，路径转换后未去除项目根目录前缀

**修复方案**:
1. `result_dir` 改为基于项目根目录的绝对路径
2. 路径转换逻辑增加 `output/vis/` 的处理分支：使用 `split('output/vis/', 1)[1]` 提取后半部分，加上 `visualize/` 前缀

**修改文件**:
- `backend/api/feature_cluster.py` 第215-217行（result_dir）、第304-314行（路径转换逻辑）

---

## 任务二：全面修复项目中硬编码相对路径

**时间**: 2026-05-30 00:25

**修复策略**: 每个文件添加模块级 `PROJECT_ROOT` 常量（如已有则复用），将所有 `os.path.join("相对目录", ...)` 替换为 `os.path.join(PROJECT_ROOT, ...)`。

### 修复的文件清单（12个文件，36处）

| 文件 | 修复数 | 修改内容 |
|------|--------|----------|
| `backend/api/few_shot.py` | 6 | uploads/few_shot, output/vis/few_shot |
| `backend/api/zero_shot.py` | 6 | uploads/zero_shot, output/vis/zero_shot |
| `backend/api/feature_cluster.py` | 6 | uploads/cluster, output/vis/cluster (含本地 project_root 改为模块级) |
| `backend/api/local_monitor.py` | 2 | output/exports |
| `backend/api/client_monitor.py` | 2 | output/exports |
| `backend/api/detection.py` | 2 | output/exports |
| `backend/core/local_monitor_service.py` | 3 | output/slices/monitor |
| `backend/core/client_monitor_service.py` | 4 | uploads/clients, 含 f-string 相对路径修复 |
| `backend/core/task_manager.py` | 2 | uploads/{task_id}, output/slices (save_dir 参数) |
| `backend/config/config_load.py` | 1 | config_path 默认值改用动态检测 |
| `backend/core/config_manager.py` | 1 | DEFAULT_CONFIG_PATH 改用静态方法动态检测 |

### 特殊修复说明

- **client_monitor_service.py**: 修复 `f"uploads/clients/segments/{base_name}.wav"` f-string 硬编码路径，同时移除下游的重复 `project_root` 拼接
- **config_load.py**: `yaml_load()` 默认参数从硬编码字符串改为 `None`，通过 `_get_default_config_path()` 动态检测
- **config_manager.py**: 用 `@staticmethod _get_default_config_path()` 替代类变量 `DEFAULT_CONFIG_PATH`
- **task_manager.py**: 在 `_import_modules()` 中存储 `self.project_root`，供其他方法使用
