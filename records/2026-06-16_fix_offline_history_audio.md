# 工作记录：修复离线检测页面历史检测结果播放问题

**时间戳**: 2026-06-16 22:30:00 (Asia/Shanghai)
**任务**: 修复离线检测页面历史检测结果无法播放的问题

## 问题描述

启动服务后，在离线检测页面恢复历史检测结果时，音频播放按钮显示但无法播放音频。

## 问题分析

通过代码审查发现以下问题：

1. **音频路径推断问题**：历史记录恢复时，`audio_slice_path` 是从 `overlay_path` 推断出来的，没有验证文件是否真实存在
2. **文件缺失处理**：如果音频文件被删除或路径不正确，播放会失败，但前端仍然显示播放按钮

### 相关代码位置

- 前端代码: `frontend/index.html`
- 历史记录加载函数: `loadOfflineResultsFromStorage()` (第1918行)
- 音频路径重建逻辑 (第1927-1941行)

## 修复方案

### 修改内容

1. **将 `loadOfflineResultsFromStorage()` 改为异步函数**
   - 支持异步验证文件是否存在

2. **新增 `validateAndCleanHistoryRecords()` 函数**
   - 使用 HEAD 请求验证音频文件和热力图文件是否存在
   - 如果音频文件不存在，将 `audio_slice_path` 设为 null
   - 如果热力图文件不存在，将相关路径设为 null
   - **如果音频和热力图文件都不存在，从离线结果数据中删除该记录**
   - 清理后保存到 localStorage

3. **更新调用链**
   - `initOfflineUpload()` 改为异步函数
   - `DOMContentLoaded` 事件处理函数改为异步函数
   - 使用 `await` 等待历史记录加载完成

### 代码变更详情

```javascript
// 1. 修改 loadOfflineResultsFromStorage 为异步函数
async function loadOfflineResultsFromStorage() {
    // ... 原有逻辑 ...
    // 验证文件是否存在并清理无效记录
    await validateAndCleanHistoryRecords();
    // ... 原有逻辑 ...
}

// 2. 新增验证和清理函数
async function validateAndCleanHistoryRecords() {
    const validRecords = [];
    const removedRecords = [];

    for (const r of offlineResultsData) {
        let hasValidFile = false;

        // 检查音频文件是否存在
        if (r.audio_slice_path) {
            try {
                const response = await fetch(`${API_BASE}/api/detection/audio/${r.audio_slice_path}`, {
                    method: 'HEAD'
                });
                if (response.ok) {
                    hasValidFile = true;
                } else {
                    r.audio_slice_path = null;
                }
            } catch (error) {
                r.audio_slice_path = null;
            }
        }

        // 检查热力图文件是否存在
        if (r.overlay_path) {
            try {
                const response = await fetch(`${API_BASE}/${r.overlay_path}`, {
                    method: 'HEAD'
                });
                if (response.ok) {
                    hasValidFile = true;
                } else {
                    r.overlay_path = null;
                    r.heatmap_path = null;
                    r.original_path = null;
                }
            } catch (error) {
                r.overlay_path = null;
                r.heatmap_path = null;
                r.original_path = null;
            }
        }

        // 如果没有任何有效文件，标记为待删除
        if (hasValidFile) {
            validRecords.push(r);
        } else {
            removedRecords.push(r.filename || '未知文件');
        }
    }

    // 更新离线结果数据，只保留有效记录
    offlineResultsData = validRecords;

    // 保存清理后的记录到 localStorage
    if (removedRecords.length > 0) {
        saveOfflineResultsToStorage();
        console.log(`[Offline] 已清理 ${removedRecords.length} 条无效历史记录`);
    }
}
```

## 验证步骤

1. 启动服务: `./start_server.sh start`
2. 访问离线检测页面: `http://localhost:8004`
3. 如果历史记录中的音频文件不存在，播放按钮不会显示
4. 如果音频文件存在，播放按钮正常显示并可播放

## 影响范围

- 仅影响离线检测页面的历史记录显示
- 不影响新的检测任务
- 不影响其他页面功能

## 后续建议

1. 考虑定期清理历史记录中指向已删除文件的条目
2. 可以添加音频文件过期提示，告知用户某些历史记录的音频已不可用
