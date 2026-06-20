// ==================== 离线检测模块 ====================

// 文件处理函数
function handleFiles(files) {
    // 追加新文件到已有列表
    const newFiles = Array.from(files);
    selectedFiles = [...selectedFiles, ...newFiles];
    updateFileList();
    log(`已添加 ${newFiles.length} 个文件，当前共 ${selectedFiles.length} 个文件`);
}

function updateFileList() {
    const list = document.getElementById('fileList');
    const dropzone = document.getElementById('dropzone');
    const fileActions = document.getElementById('fileActions');

    if (selectedFiles.length === 0) {
        list.innerHTML = '';
        dropzone.style.display = 'block';
        fileActions.style.display = 'none';
        return;
    }

    // 有文件时隐藏上传框，显示操作按钮
    dropzone.style.display = 'none';
    fileActions.style.display = 'block';

    // 显示所有文件（通过CSS限制显示高度，支持滚动）
    list.innerHTML = selectedFiles.map((f, index) => `
        <div class="file-item">
            <span>📄 ${f.name}</span>
            <span>${(f.size / 1024 / 1024).toFixed(2)} MB</span>
        </div>
    `).join('');
}

function clearFiles() {
    selectedFiles = [];
    document.getElementById('fileInput').value = '';
    updateFileList();
    log('文件列表已清除');
}

function continueUpload() {
    // 清空 input 值，允许重复选择相同文件
    document.getElementById('fileInput').value = '';
    // 触发文件选择框
    document.getElementById('fileInput').click();
}

// 日志输出
function log(message, type = 'info') {
    const logBox = document.getElementById('statusLog');
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.innerHTML = `<span class="timestamp">[${time}]</span> ${message}`;
    logBox.appendChild(line);
    logBox.scrollTop = logBox.scrollHeight;
}

// WebSocket 连接
function connectWebSocket(taskId) {
    const wsUrl = `ws://${window.location.host}/ws/progress/${taskId}`;
    wsConnection = new WebSocket(wsUrl);

    wsConnection.onopen = () => {
        console.log('WebSocket connected');
    };

    wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    wsConnection.onerror = (error) => {
        console.error('WebSocket error:', error);
        log('WebSocket 连接错误', 'error');
        // WebSocket 错误时重置状态
        isDetecting = false;
        const btn = document.getElementById('startBtn');
        btn.disabled = false;
        btn.textContent = '🚀 开始检测';
    };

    wsConnection.onclose = () => {
        console.log('WebSocket closed');
        // 如果检测状态仍为 true，说明是异常关闭，重置状态
        if (isDetecting) {
            isDetecting = false;
            const btn = document.getElementById('startBtn');
            btn.disabled = false;
            btn.textContent = '🚀 开始检测';
            log('连接已断开', 'error');
        }
    };
}

function handleWebSocketMessage(data) {
    if (data.type === 'progress') {
        const progress = data.data;
        updateProgress(progress.progress, progress.status);

        // 根据状态显示详细日志
        if (progress.status === 'preprocessing' && progress.message) {
            log(progress.message);
        } else if (progress.status === 'detecting' && progress.message) {
            log(progress.message);
        } else if (progress.current_file) {
            log(`处理: ${progress.current_file}`);
        }

        // 显示统计信息
        if (progress.current !== undefined && progress.total !== undefined) {
            log(`进度: ${progress.current}/${progress.total}`);
        }
    } else if (data.type === 'result') {
        const result = data.data;
        // 保存任务ID用于导出
        if (result.task_id) {
            currentOfflineTaskId = result.task_id;
        }
        if (result.status === 'completed') {
            log(`✅ 检测完成! 共处理 ${result.results?.length || 0} 个文件`);
            const anomalyCount = result.results?.filter(r => r.is_anomaly).length || 0;
            const failedCount = result.results?.filter(r => r.status === '预处理失败').length || 0;
            if (anomalyCount > 0) {
                log(`⚠️ 发现 ${anomalyCount} 个异常文件`);
            }
            if (failedCount > 0) {
                log(`⚠️ ${failedCount} 个文件预处理失败`);
                // 显示详细的失败原因
                const failedFiles = result.results?.filter(r => r.status === '预处理失败') || [];
                failedFiles.forEach(f => {
                    log(`   ❌ ${f.filename}: ${f.error || '预处理失败'}`);
                });
            }
        } else if (result.status === 'failed') {
            log(`❌ 检测失败: ${result.error}`, 'error');
        }
        showResults(result);
        wsConnection.close();

        // 检测完成后启用按钮
        isDetecting = false;
        const btn = document.getElementById('startBtn');
        btn.disabled = false;
        btn.textContent = '🚀 开始检测';
    }
}

function updateProgress(percent, status) {
    const progress = document.getElementById('progress');
    const fill = document.getElementById('progressFill');
    const text = document.getElementById('progressText');

    progress.style.display = 'block';
    fill.style.width = `${percent}%`;

    const statusMap = {
        'preprocessing': '预处理中...',
        'detecting': '异常检测中...',
        'completed': '完成!'
    };
    text.textContent = statusMap[status] || `${percent.toFixed(1)}%`;
}

// 开始检测
async function startDetection() {
    if (selectedFiles.length === 0) {
        await showModal('请先选择音频文件', '提示');
        return;
    }

    // 检查是否已有检测任务在进行中
    if (isDetecting) {
        await showModal('检测任务正在进行中，请等待当前任务完成后再试', '提示');
        return;
    }

    const btn = document.getElementById('startBtn');
    isDetecting = true;
    btn.disabled = true;
    btn.textContent = '检测中...';

    // 显示上传信息
    const totalSize = selectedFiles.reduce((sum, f) => sum + f.size, 0);
    log(`📁 准备上传 ${selectedFiles.length} 个文件，共 ${(totalSize / 1024 / 1024).toFixed(2)} MB`);

    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('files', f));
    formData.append('algorithm', document.getElementById('algorithm').value);
    formData.append('device', document.getElementById('device').value);

    // 系统自动匹配参考音频，不传递reference_audio参数
    log(`🎵 系统将自动从参考音频库中匹配`);

    log('⬆️ 正在上传文件...');

    try {
        const response = await fetch(`${API_BASE}/api/detection/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            currentTaskId = result.task_id;
            log(`✅ 任务创建成功: ${currentTaskId.substring(0, 8)}...`);
            if (result.queue_position > 0) {
                log(`⏳ 队列位置: ${result.queue_position}，请稍候...`);
            } else {
                log('🚀 开始处理...');
            }

            // 连接 WebSocket
            connectWebSocket(currentTaskId);
        } else {
            log(`❌ 错误: ${result.detail}`, 'error');
            // 请求失败时重置状态
            isDetecting = false;
            btn.disabled = false;
            btn.textContent = '🚀 开始检测';
        }
    } catch (error) {
        log(`❌ 请求失败: ${error.message}`, 'error');
        // 请求失败时重置状态
        isDetecting = false;
        btn.disabled = false;
        btn.textContent = '🚀 开始检测';
    }
    // 注意：任务成功创建后，按钮保持禁用状态，直到收到结果
}

// 存储离线检测结果用于视图切换
let offlineResultsData = [];

// 离线检测结果 localStorage 键名
const OFFLINE_RESULTS_KEY = 'offline_detection_results';

// 从 localStorage 加载离线检测历史结果
async function loadOfflineResultsFromStorage() {
    try {
        const stored = localStorage.getItem(OFFLINE_RESULTS_KEY);
        if (stored) {
            const data = JSON.parse(stored);
            offlineResultsData = data.results || [];
            console.log(`[Offline] 从本地存储加载了 ${offlineResultsData.length} 条历史结果`);

            // 为历史记录重建 audio_slice_path（如果缺失）
            let rebuiltCount = 0;
            offlineResultsData.forEach(r => {
                if (!r.audio_slice_path && r.overlay_path) {
                    // 从 overlay_path 推断音频路径
                    // 注意：overlay_path 文件名包含 _overlay 后缀，但音频文件没有
                    const baseName = r.overlay_path.replace(/^.*[\\/]/, '').replace(/\.[^/.]+$/, '').replace(/_overlay$/, '');
                    r.audio_slice_path = `slice/audio/${baseName}.wav`;
                    rebuiltCount++;
                    console.log(`[Offline] 为历史记录重建音频路径: ${r.filename} -> ${r.audio_slice_path}`);
                }
            });
            if (rebuiltCount > 0) {
                console.log(`[Offline] 为 ${rebuiltCount} 条历史记录重建了音频路径`);
            }

            // 验证文件是否存在并清理无效记录
            await validateAndCleanHistoryRecords();

            // 恢复显示
            if (offlineResultsData.length > 0) {
                restoreOfflineResultsDisplay();
                log(`📂 已恢复 ${offlineResultsData.length} 条历史检测结果`, 'success');
            }
        }
    } catch (error) {
        console.error('[Offline] 加载历史结果失败:', error);
        offlineResultsData = [];
    }
}

// 验证历史记录中的文件是否存在，清理无效记录
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
                    console.log(`[Offline] 音频文件不存在: ${r.audio_slice_path}`);
                    r.audio_slice_path = null;
                }
            } catch (error) {
                console.log(`[Offline] 验证音频文件失败: ${r.audio_slice_path}`, error);
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
                    console.log(`[Offline] 热力图文件不存在: ${r.overlay_path}`);
                    r.overlay_path = null;
                    r.heatmap_path = null;
                    r.original_path = null;
                }
            } catch (error) {
                console.log(`[Offline] 验证热力图文件失败: ${r.overlay_path}`, error);
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
        console.log(`[Offline] 已清理 ${removedRecords.length} 条无效历史记录:`, removedRecords);
    }

    console.log(`[Offline] 历史记录验证完成，有效记录: ${validRecords.length} 条`);
}

// 保存离线检测结果到 localStorage
function saveOfflineResultsToStorage() {
    try {
        const data = {
            status: 'completed',
            results: offlineResultsData,
            timestamp: new Date().toISOString()
        };
        localStorage.setItem(OFFLINE_RESULTS_KEY, JSON.stringify(data));
    } catch (error) {
        console.error('[Offline] 保存结果到本地存储失败:', error);
    }
}

// 清空离线检测结果 localStorage
function clearOfflineResultsStorage() {
    try {
        localStorage.removeItem(OFFLINE_RESULTS_KEY);
    } catch (error) {
        console.error('[Offline] 清空本地存储失败:', error);
    }
}

// 恢复离线检测结果显示
function restoreOfflineResultsDisplay() {
    const resultsDiv = document.getElementById('results');
    const exportBtn = document.getElementById('exportOfflineBtn');
    const clearBtn = document.getElementById('clearOfflineBtn');

    // 显示导出按钮（在检测结果卡片标题中）
    if (exportBtn) exportBtn.style.display = 'inline-flex';
    // 显示清空按钮（在检测设置卡片中，与开始检测并排）
    if (clearBtn) clearBtn.style.display = 'inline-block';

    const totalCount = offlineResultsData.length;
    const anomalyCount = offlineResultsData.filter(r => r.is_anomaly).length;
    const failedCount = offlineResultsData.filter(r => r.status === '预处理失败').length;
    const normalCount = totalCount - anomalyCount - failedCount;

    // 生成摘要
    let summaryHtml = `
        <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <h3 style="margin: 0 0 12px 0; font-size: 16px;">📊 检测摘要</h3>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; font-size: 14px;">
                <div><strong>总文件数:</strong> ${totalCount}</div>
                <div><strong>异常数:</strong> <span style="color: #dc3545;">${anomalyCount}</span></div>
                <div><strong>正常数:</strong> <span style="color: #28a745;">${normalCount}</span></div>
                <div><strong>预处理失败:</strong> <span style="color: #ffc107;">${failedCount}</span></div>
                <div><strong>检测时间:</strong> <span style="color: #666;">历史记录</span></div>
            </div>
        </div>
    `;

    // 生成结果卡片
    let resultsHtml = `<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px;">`;

    for (let i = offlineResultsData.length - 1; i >= 0; i--) {
        const r = offlineResultsData[i];
        const uniqueId = `offline-${i}`;
        const isAnomaly = r.is_anomaly;
        const isFailed = r.status === '预处理失败';
        const isNoMatch = r.status === '未匹配';

        // 调试：输出音频路径信息
        console.log(`[Debug] 结果 ${i}: filename=${r.filename}, audio_slice_path=${r.audio_slice_path}, overlay_path=${r.overlay_path}`);
        if (!r.audio_slice_path && r.overlay_path) {
            console.log(`[Debug] 警告: 有overlay_path但没有audio_slice_path，可能是历史记录`);
        }

        let statusColor, statusBg, statusText;
        if (isFailed || isNoMatch) {
            statusColor = '#856404';
            statusBg = '#fff3cd';
            statusText = isFailed ? '⚠️ 预处理失败' : '⚠️ 未匹配';
        } else if (isAnomaly) {
            statusColor = '#dc3545';
            statusBg = '#f8d7da';
            statusText = '🔴 异常';
        } else {
            statusColor = '#28a745';
            statusBg = '#d4edda';
            statusText = '🟢 正常';
        }

        const hasHeatmap = r.heatmap_path !== null && r.heatmap_path !== undefined;
        const hasOriginal = r.original_path !== null && r.original_path !== undefined;
        const hasOverlay = r.overlay_path !== null && r.overlay_path !== undefined;

        const displayUrl = hasOverlay ? `${API_BASE}/${r.overlay_path}` :
                          (hasHeatmap ? `${API_BASE}/${r.heatmap_path}` : '');
        const originalUrl = hasOriginal ? `${API_BASE}/${r.original_path}` : '';
        const overlayUrl = hasOverlay ? `${API_BASE}/${r.overlay_path}` :
                          (hasHeatmap ? `${API_BASE}/${r.heatmap_path}` : '');
        const heatmapUrl = hasHeatmap ? `${API_BASE}/${r.heatmap_path}` : '';

        const timestamp = r.timestamp ? new Date(r.timestamp).toLocaleString() : '未知时间';

        resultsHtml += `
            <div style="border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: white;">
                <div style="padding: 12px; border-bottom: 1px solid #e0e0e0; background: #f8f9fa; display: flex; align-items: center; gap: 8px;">
                    <div style="font-weight: 600; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;" title="${r.filename}${r.segment_index !== undefined ? ` #${r.segment_index} @${r.segment_start?.toFixed(2)}s` : ''}">
                        🎵 ${r.filename}
                        ${r.segment_index !== undefined ? `<span style="color:#999;font-weight:400;font-size:11px;margin-left:4px;">#${r.segment_index} @${(r.segment_start || 0).toFixed(2)}s</span>` : ''}
                    </div>
                    ${r.audio_slice_path ? `
                        <button onclick="toggleAudioPlay('${r.audio_slice_path}', this)"
                                class="audio-play-btn"
                                title="点击试听"
                                style="width: 22px; height: 22px; border: none; background: #6366f1; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 0; transition: all 0.2s; outline: none; flex-shrink: 0;"
                                onmouseover="this.style.background='#4f46e5'"
                                onmouseout="this.style.background='#6366f1'"
                                onfocus="this.style.boxShadow='0 0 0 2px rgba(99, 102, 241, 0.5)'"
                                onblur="this.style.boxShadow='none'">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="white">
                                <polygon points="5,3 19,12 5,21"/>
                            </svg>
                        </button>
                    ` : ''}
                </div>
                <div style="padding: 12px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; font-size: 12px;">
                        <div><strong>异常分数:</strong> ${r.anomaly_score ? r.anomaly_score.toFixed(4) : 'N/A'}</div>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <strong>状态:</strong>
                            <span style="background: ${statusBg}; color: ${statusColor}; padding: 2px 6px; border-radius: 4px; font-size: 11px;">
                                ${statusText}
                            </span>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; font-size: 12px; flex-wrap: wrap;">
                        <span><strong>序号:</strong> #${offlineResultsData.length - i}</span>
                        <span><strong>参考音频:</strong> ${r.music_name || (r.segment_info && r.segment_info.music_name) || '未匹配'}</span>
                    </div>
                    ${hasHeatmap ? `
                        <div style="margin-top: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <div style="font-size: 11px; color: #666;">🔥 异常热力图:</div>
                                <div style="display: flex; gap: 4px;">
                                    <button id="btn-offline-original-${uniqueId}"
                                            onclick="toggleOfflineHeatmapMode('${uniqueId}', 'original')"
                                            style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">原图</button>
                                    <button id="btn-offline-overlay-${uniqueId}"
                                            onclick="toggleOfflineHeatmapMode('${uniqueId}', 'overlay')"
                                            style="padding: 2px 8px; font-size: 10px; border: 1px solid #667eea; background: #667eea; color: white; border-radius: 4px; cursor: pointer;">叠加</button>
                                    <button id="btn-offline-heatmap-${uniqueId}"
                                            onclick="toggleOfflineHeatmapMode('${uniqueId}', 'heatmap')"
                                            style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">纯热力图</button>
                                </div>
                            </div>
                            <img id="offline-heatmap-img-${uniqueId}"
                                 src="${displayUrl}"
                                 data-original="${originalUrl}"
                                 data-overlay="${overlayUrl}"
                                 data-heatmap="${heatmapUrl}"
                                 style="width: 100%; border-radius: 4px; border: 1px solid #e0e0e0; cursor: pointer;"
                                 onclick="showHeatmapModal('${overlayUrl}', '${r.filename}', ${r.anomaly_score || 0})"
                                 title="点击查看大图">
                        </div>
                    ` : isNoMatch ? `
                        <div style="margin-top: 8px; padding: 20px; background: #fff3cd; border-radius: 4px; text-align: center; color: #856404; font-size: 12px;">
                            ⚠️ 未找到匹配的参考音频片段
                        </div>
                    ` : `
                        <div style="margin-top: 8px; padding: 20px; background: #f5f5f5; border-radius: 4px; text-align: center; color: #999; font-size: 12px;">
                            暂无热力图
                        </div>
                    `}
                </div>
            </div>
        `;
    }

    resultsHtml += `</div>`;
    resultsDiv.innerHTML = summaryHtml + resultsHtml;
}

// 显示结果
function showResults(data) {
    const resultsDiv = document.getElementById('results');
    const heatmapCard = document.getElementById('heatmapCard');
    const exportBtn = document.getElementById('exportOfflineBtn');

    // 显示导出按钮（在检测结果卡片标题中）
    if (data.status === 'completed' && exportBtn) {
        exportBtn.style.display = 'inline-flex';
    }
    // 显示清空按钮（在检测设置卡片中，与开始检测并排）
    const clearOfflineBtn = document.getElementById('clearOfflineBtn');
    if (data.status === 'completed' && clearOfflineBtn) {
        clearOfflineBtn.style.display = 'inline-block';
    }

    // 隐藏旧的热力图卡片（使用新的统一展示方式）
    if (heatmapCard) {
        heatmapCard.style.display = 'none';
    }

    if (data.status === 'completed' && data.results) {
        const totalCount = data.results.length;
        const anomalyCount = data.results.filter(r => r.is_anomaly).length;
        const failedCount = data.results.filter(r => r.status === '预处理失败').length;
        const normalCount = totalCount - anomalyCount - failedCount;

        // 保存结果数据
        offlineResultsData = data.results;

        // 保存到 localStorage
        saveOfflineResultsToStorage();

        // 生成摘要
        let summaryHtml = `
            <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                <h3 style="margin: 0 0 12px 0; font-size: 16px;">📊 检测摘要</h3>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; font-size: 14px;">
                    <div><strong>总文件数:</strong> ${totalCount}</div>
                    <div><strong>异常数:</strong> <span style="color: #dc3545;">${anomalyCount}</span></div>
                    <div><strong>正常数:</strong> <span style="color: #28a745;">${normalCount}</span></div>
                    <div><strong>预处理失败:</strong> <span style="color: #ffc107;">${failedCount}</span></div>
                    <div><strong>算法:</strong> ${data.algorithm || '未知'}</div>
                </div>
            </div>
        `;

        // 生成卡片式详细结果
        let resultsHtml = `
            <h4 style="margin: 16px 0 12px 0; font-size: 15px;">📋 详细结果</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px;">
        `;

        data.results.forEach((r, index) => {
            const uniqueId = `offline-${index}`;
            const isFailed = r.status === '预处理失败';
            const isAnomaly = r.is_anomaly;

            // 预处理失败的文件显示特殊样式
            if (isFailed) {
                resultsHtml += `
                    <div style="border: 1px solid #ffc107; border-radius: 8px; overflow: hidden; background: #fffbeb;">
                        <div style="padding: 12px; border-bottom: 1px solid #ffc107; background: #fef3c7;">
                            <div style="font-weight: 600; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${r.filename}">
                                🎵 ${r.filename}
                            </div>
                        </div>
                        <div style="padding: 12px;">
                            <div style="display: grid; grid-template-columns: 1fr; gap: 8px; margin-bottom: 12px; font-size: 12px;">
                                <div><strong>状态:</strong>
                                    <span style="background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-size: 11px;">
                                        ⚠️ 预处理失败
                                    </span>
                                </div>
                                <div style="color: #92400e; font-size: 11px; margin-top: 4px;">
                                    <strong>原因:</strong> ${r.error || '未知错误'}
                                </div>
                                <div><strong>序号:</strong> #${index + 1}</div>
                            </div>
                            <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
                                该文件未能成功预处理，无法生成热力图
                            </div>
                        </div>
                    </div>
                `;
                return; // 跳过正常的结果卡片生成
            }

            const statusColor = isAnomaly ? '#dc3545' : '#28a745';
            const statusBg = isAnomaly ? '#f8d7da' : '#d4edda';
            const hasHeatmap = r.heatmap_path !== null && r.heatmap_path !== undefined;
            const hasOriginal = r.original_path !== null && r.original_path !== undefined;
            const hasOverlay = r.overlay_path !== null && r.overlay_path !== undefined;

            // 使用叠加图作为默认显示，如果没有则使用热力图
            const displayUrl = hasOverlay ? `${API_BASE}/${r.overlay_path}` :
                              (hasHeatmap ? `${API_BASE}/${r.heatmap_path}` : '');
            const originalUrl = hasOriginal ? `${API_BASE}/${r.original_path}` : '';
            const overlayUrl = hasOverlay ? `${API_BASE}/${r.overlay_path}` :
                              (hasHeatmap ? `${API_BASE}/${r.heatmap_path}` : '');
            const heatmapUrl = hasHeatmap ? `${API_BASE}/${r.heatmap_path}` : '';

            resultsHtml += `
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: white;">
                    <div style="padding: 12px; border-bottom: 1px solid #e0e0e0; background: #f8f9fa; display: flex; align-items: center; gap: 8px;">
                        <div style="font-weight: 600; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;" title="${r.filename}">
                            🎵 ${r.filename}
                        </div>
                        ${r.audio_slice_path ? `
                            <button onclick="toggleAudioPlay('${r.audio_slice_path}', this)"
                                    title="点击试听"
                                    style="width: 22px; height: 22px; border: none; background: #6366f1; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center; padding: 0; transition: all 0.2s; outline: none; flex-shrink: 0;"
                                    onmouseover="this.style.background='#4f46e5'"
                                    onmouseout="this.style.background='#6366f1'"
                                    onfocus="this.style.boxShadow='0 0 0 2px rgba(99, 102, 241, 0.5)'"
                                    onblur="this.style.boxShadow='none'">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="white">
                                    <polygon points="5,3 19,12 5,21"/>
                                </svg>
                            </button>
                        ` : ''}
                    </div>
                    <div style="padding: 12px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; font-size: 12px;">
                            <div><strong>异常分数:</strong> ${r.anomaly_score.toFixed(4)}</div>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <strong>状态:</strong>
                                <span style="background: ${statusBg}; color: ${statusColor}; padding: 2px 6px; border-radius: 4px; font-size: 11px;">
                                    ${isAnomaly ? '🔴 异常' : '🟢 正常'}
                                </span>
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; font-size: 12px; flex-wrap: wrap;">
                            <span><strong>序号:</strong> #${index + 1}</span>
                            <span><strong>参考音频:</strong> ${r.music_name || '未匹配'}</span>
                        </div>
                        ${hasHeatmap ? `
                            <div style="margin-top: 8px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <div style="font-size: 11px; color: #666;">🔥 异常热力图:</div>
                                    <div style="display: flex; gap: 4px;">
                                        <button id="btn-offline-original-${uniqueId}"
                                                onclick="toggleOfflineHeatmapMode('${uniqueId}', 'original')"
                                                style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">原图</button>
                                        <button id="btn-offline-overlay-${uniqueId}"
                                                onclick="toggleOfflineHeatmapMode('${uniqueId}', 'overlay')"
                                                style="padding: 2px 8px; font-size: 10px; border: 1px solid #667eea; background: #667eea; color: white; border-radius: 4px; cursor: pointer;">叠加</button>
                                        <button id="btn-offline-heatmap-${uniqueId}"
                                                onclick="toggleOfflineHeatmapMode('${uniqueId}', 'heatmap')"
                                                style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">纯热力图</button>
                                    </div>
                                </div>
                                <img id="offline-heatmap-img-${uniqueId}"
                                     src="${displayUrl}"
                                     data-original="${originalUrl}"
                                     data-overlay="${overlayUrl}"
                                     data-heatmap="${heatmapUrl}"
                                     style="width: 100%; border-radius: 4px; border: 1px solid #e0e0e0; cursor: pointer;"
                                     onclick="showHeatmapModal('${overlayUrl}', '${r.filename}', ${r.anomaly_score})"
                                     title="点击查看大图">
                            </div>
                        ` : `
                            <div style="margin-top: 8px; padding: 20px; background: #f5f5f5; border-radius: 4px; text-align: center; color: #999; font-size: 12px;">
                                暂无热力图
                            </div>
                        `}
                    </div>
                </div>
            `;
        });

        resultsHtml += `</div>`;
        resultsDiv.innerHTML = summaryHtml + resultsHtml;

    } else if (data.status === 'failed') {
        resultsDiv.innerHTML = `<div class="alert alert-error">检测失败: ${data.error}</div>`;
        if (heatmapCard) heatmapCard.style.display = 'none';
    }
}

// 离线检测热力图模式切换
function toggleOfflineHeatmapMode(uniqueId, mode) {
    const img = document.getElementById(`offline-heatmap-img-${uniqueId}`);
    const btnOriginal = document.getElementById(`btn-offline-original-${uniqueId}`);
    const btnOverlay = document.getElementById(`btn-offline-overlay-${uniqueId}`);
    const btnHeatmap = document.getElementById(`btn-offline-heatmap-${uniqueId}`);

    if (!img || !btnOverlay || !btnHeatmap || !btnOriginal) return;

    // 重置所有按钮样式
    const inactiveStyle = { background: 'white', color: '#666', borderColor: '#d9d9d9' };
    const activeStyle = { background: '#667eea', color: 'white', borderColor: '#667eea' };

    Object.assign(btnOriginal.style, inactiveStyle);
    Object.assign(btnOverlay.style, inactiveStyle);
    Object.assign(btnHeatmap.style, inactiveStyle);

    if (mode === 'original') {
        if (!img.dataset.original) {
            showModal('原图未成功保存，请检查后端日志', '提示');
            return;
        }
        img.src = img.dataset.original;
        Object.assign(btnOriginal.style, activeStyle);
    } else if (mode === 'overlay') {
        img.src = img.dataset.overlay;
        Object.assign(btnOverlay.style, activeStyle);
    } else {
        img.src = img.dataset.heatmap;
        Object.assign(btnHeatmap.style, activeStyle);
    }
}

// 当前离线检测任务ID（用于导出）
let currentOfflineTaskId = null;

// 导出离线检测结果
async function exportOfflineResults() {
    if (!currentOfflineTaskId) {
        await showModal('暂无检测结果可导出', '提示');
        return;
    }

    try {
        const btn = document.getElementById('exportOfflineBtn');
        btn.textContent = '⏳ 导出中...';
        btn.disabled = true;

        const response = await fetch(`${API_BASE}/api/detection/export/${currentOfflineTaskId}`);

        if (response.ok) {
            // 获取文件名
            const contentDisposition = response.headers.get('content-disposition');
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
            let filename = `检测结果_${timestamp}.zip`;
            if (contentDisposition) {
                const match = contentDisposition.match(/filename="?(.+?)"?$/);
                if (match) filename = match[1];
            }

            // 下载文件
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            log('✅ 结果导出成功');
        } else {
            const error = await response.json();
            await showModal(`导出失败: ${error.detail}`, '错误');
        }
    } catch (error) {
        console.error('导出失败:', error);
        await showModal('导出失败，请重试', '错误');
    } finally {
        const btn = document.getElementById('exportOfflineBtn');
        btn.textContent = '📥 导出结果';
        btn.disabled = false;
    }
}

// 清空离线检测结果
function clearOfflineResults() {
    offlineResultsData = [];

    // 清空 localStorage
    clearOfflineResultsStorage();

    // 重置显示
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div style="text-align: center; color: #999; padding: 40px;">
            等待检测...
        </div>
    `;

    // 隐藏导出按钮和清空按钮
    const exportBtn = document.getElementById('exportOfflineBtn');
    const clearBtn = document.getElementById('clearOfflineBtn');
    if (exportBtn) exportBtn.style.display = 'none';
    if (clearBtn) clearBtn.style.display = 'none';

    // 重置当前任务ID
    currentOfflineTaskId = null;

    log('🗑️ 已清空所有离线检测结果', 'success');
}

// 离线检测文件上传初始化
async function initOfflineUpload() {
    // 从 localStorage 加载历史结果
    await loadOfflineResultsFromStorage();

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');

    if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        console.log('[Offline] 文件上传事件监听器已绑定');
    } else {
        console.error('[Offline] 找不到上传区域元素:', { dropzone: !!dropzone, fileInput: !!fileInput });
    }
}
