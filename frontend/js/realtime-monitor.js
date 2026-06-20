// ==================== 实时监控模块 (当前隐藏) ====================

function monitorLog(message) {
    const logBox = document.getElementById('monitorLog');
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.innerHTML = `<span class="timestamp">[${time}]</span> ${message}`;
    logBox.appendChild(line);
    logBox.scrollTop = logBox.scrollHeight;
}

async function startMonitor() {
    const dir = document.getElementById('monitorDir').value;
    if (!dir) {
        await showModal('请输入监控目录', '提示');
        return;
    }

    const selectedAlgorithm = document.getElementById('monitorAlgorithm').value;
    const selectedDevice = document.getElementById('monitorDevice').value;

    // 先检查检测上下文
    try {
        const contextResponse = await fetch(`${API_BASE}/api/monitor/detection-context`);
        const context = await contextResponse.json();

        // 如果有运行中的任务，检查算法和设备是否匹配
        if (context.has_running_task) {
            let mismatchMessage = '';

            if (context.current_algorithm && context.current_algorithm !== selectedAlgorithm) {
                mismatchMessage += `• 离线检测使用算法: ${context.current_algorithm}\n`;
                mismatchMessage += `• 监控选择算法: ${selectedAlgorithm}\n\n`;
            }

            if (context.current_device && context.current_device !== selectedDevice) {
                mismatchMessage += `• 离线检测使用设备: ${context.current_device}\n`;
                mismatchMessage += `• 监控选择设备: ${selectedDevice}\n\n`;
            }

            if (mismatchMessage) {
                const fullMessage = `检测到运行中的离线检测任务，配置不匹配！\n\n${mismatchMessage}为保证检测准确性，监控必须使用与离线检测相同的算法和设备。\n\n请选择以下操作：`;

                const action = await showMismatchModal(fullMessage, context.current_algorithm, context.current_device);

                if (action === 'cancel') {
                    return;
                } else if (action === 'sync') {
                    // 自动同步算法和设备
                    document.getElementById('monitorAlgorithm').value = context.current_algorithm;
                    document.getElementById('monitorDevice').value = context.current_device;
                    monitorLog(`已自动同步为离线检测配置: 算法=${context.current_algorithm}, 设备=${context.current_device}`);
                    // 继续执行启动
                }
                // 如果 action === 'force'，继续执行启动（后端会再次检查并阻止）
            }
        }
    } catch (error) {
        console.error('检查检测上下文失败:', error);
    }

    // 获取选择的参考音频
    const selectedRefs = monitorSelectedReferences.map(r => r.path);

    // 启动监控
    const config = {
        directory: dir,
        algorithm: document.getElementById('monitorAlgorithm').value,
        device: document.getElementById('monitorDevice').value,
        detect_existing: false,
        reference_audios: selectedRefs
    };

    monitorLog(`🎵 参考音频: ${monitorSelectedReferences.length > 0 ? monitorSelectedReferences.map(r => r.name).join(', ') : '自动匹配'}`);

    try {
        const response = await fetch(`${API_BASE}/api/monitor/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();

        if (response.ok) {
            monitorLog('监控已启动');
            document.getElementById('monitorStatus').textContent = '运行中';
            document.getElementById('startMonitorBtn').disabled = true;
            document.getElementById('stopMonitorBtn').disabled = false;

            // 禁用算法和设备选择
            document.getElementById('monitorAlgorithm').disabled = true;
            document.getElementById('monitorDevice').disabled = true;

            // 连接监控 WebSocket
            connectMonitorWebSocket();
        } else {
            // 处理 409 冲突错误
            if (response.status === 409) {
                await showModal(result.detail, '配置冲突', 'alert');
            } else {
                monitorLog(`启动失败: ${result.detail}`);
            }
        }
    } catch (error) {
        monitorLog(`错误: ${error.message}`);
    }
}

async function stopMonitor() {
    try {
        const response = await fetch(`${API_BASE}/api/monitor/stop`, {
            method: 'POST'
        });

        if (response.ok) {
            monitorLog('监控已停止');
            document.getElementById('monitorStatus').textContent = '停止';
            document.getElementById('startMonitorBtn').disabled = false;
            document.getElementById('stopMonitorBtn').disabled = true;

            // 启用算法和设备选择
            document.getElementById('monitorAlgorithm').disabled = false;
            document.getElementById('monitorDevice').disabled = false;

            if (monitorWs) {
                monitorWs.close();
            }

            // 停止监控后显示汇总信息
            const total = document.getElementById('monitorTotal').textContent;
            const anomaly = document.getElementById('monitorAnomaly').textContent;
            monitorLog(`监控结束，共检测 ${total} 个文件，发现 ${anomaly} 个异常`);
        }
    } catch (error) {
        monitorLog(`错误: ${error.message}`);
    }
}

function connectMonitorWebSocket() {
    // 使用广播通道接收监控更新
    const wsUrl = `ws://${window.location.host}/ws/progress/broadcast`;
    monitorWs = new WebSocket(wsUrl);

    monitorWs.onopen = () => {
        console.log('[Monitor] WebSocket connected');
        monitorLog('WebSocket 已连接');
    };

    monitorWs.onmessage = (event) => {
        console.log('[Monitor] Received:', event.data);
        const data = JSON.parse(event.data);
        if (data.type === 'monitor_update') {
            updateMonitorStats(data.data);
        } else if (data.type === 'monitor_log') {
            // 处理监控日志消息
            const logData = data.data;
            monitorLog(logData.message);
        } else if (data.type === 'monitor_config_update') {
            // 处理配置变更通知（多客户端同步）
            handleMonitorConfigUpdate(data.data);
        }
    };

    monitorWs.onerror = (error) => {
        console.error('[Monitor] WebSocket error:', error);
        monitorLog('WebSocket 错误');
    };

    monitorWs.onclose = () => {
        console.log('[Monitor] WebSocket closed');
    };
}

// 更新监控参考音频（运行时生效）
async function updateMonitorReferences() {
    const selectedRefs = monitorSelectedReferences.map(r => r.path);

    monitorLog(`🔄 正在更新参考音频...`);
    monitorLog(`🎵 当前选择: ${monitorSelectedReferences.length > 0 ? monitorSelectedReferences.map(r => r.name).join(', ') : '自动匹配'}`);

    try {
        const response = await fetch(`${API_BASE}/api/monitor/update-references`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reference_audios: selectedRefs
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            monitorLog(`✅ ${result.message}`);
            if (result.added && result.added.length > 0) {
                result.added.forEach(item => {
                    monitorLog(`   + 已添加: ${item.name}`);
                });
            }
            if (result.failed && result.failed.length > 0) {
                result.failed.forEach(item => {
                    monitorLog(`   ⚠️ 失败: ${item.path} (${item.reason})`);
                });
            }
        } else {
            monitorLog(`❌ 更新失败: ${result.message || '未知错误'}`);
        }
    } catch (error) {
        monitorLog(`❌ 错误: ${error.message}`);
    }
}

// 处理监控配置变更通知（多客户端同步）
async function handleMonitorConfigUpdate(configData) {
    console.log('[Monitor] 收到配置变更通知:', configData);

    if (configData.config_type === 'reference_audios') {
        const newRefs = configData.reference_audios || [];

        // 检查是否是当前页面发起的变更（通过时间戳简单判断，避免重复更新）
        // 实际应用中可以使用更精确的方式，如页面ID

        // 从参考音频列表中查找对应的名称
        const refsWithNames = [];
        for (const refPath of newRefs) {
            // 尝试从已加载的参考音频列表中查找名称
            const select = document.getElementById('monitorReferenceAudioSelect');
            let refName = null;

            for (let i = 0; i < select.options.length; i++) {
                if (select.options[i].value === refPath) {
                    refName = select.options[i].dataset.name || select.options[i].textContent.split(' (')[0];
                    break;
                }
            }

            // 如果没找到，使用文件名作为名称
            if (!refName) {
                refName = refPath.split('/').pop().split('\\').pop().replace(/\.[^/.]+$/, '');
            }

            refsWithNames.push({ path: refPath, name: refName });
        }

        // 更新本地参考音频列表
        monitorSelectedReferences = refsWithNames;

        // 更新UI显示
        updateMonitorSelectedRefsDisplay();

        // 显示同步提示
        if (refsWithNames.length > 0) {
            monitorLog(`🔄 [同步] 参考音频已更新: ${refsWithNames.map(r => r.name).join(', ')}`);
        } else {
            monitorLog(`🔄 [同步] 参考音频已清空，使用自动匹配`);
        }

        console.log('[Monitor] 参考音频已同步:', monitorSelectedReferences);
    }
}

// 存储所有监控结果（用于显示热力图）
let monitorResults = [];

async function exportMonitorResults() {
    try {
        const response = await fetch(`${API_BASE}/api/monitor/export`);

        if (response.ok) {
            // 获取文件名
            const contentDisposition = response.headers.get('content-disposition');
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
            let filename = `监控结果_${timestamp}.zip`;
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

            monitorLog('✅ 监控结果导出成功');
        } else {
            const error = await response.json();
            await showModal(`导出失败: ${error.detail}`, '错误');
        }
    } catch (error) {
        console.error('导出失败:', error);
        await showModal('导出失败，请重试', '错误');
    }
}

// 实时监控结果计数器
let monitorResultIdCounter = 0;

function updateMonitorStats(data) {
    document.getElementById('monitorTotal').textContent = data.total_processed;
    document.getElementById('monitorAnomaly').textContent = data.anomaly_count;

    if (data.latest_result) {
        // 保存结果用于显示热力图
        monitorResults.push(data.latest_result);
        // 更新卡片式展示
        updateMonitorResultsDisplay();
    }
}

function updateMonitorResultsDisplay() {
    const resultsDiv = document.getElementById('monitorResults');

    if (monitorResults.length === 0) {
        resultsDiv.innerHTML = `
            <div style="text-align: center; color: #999; padding: 40px;">
                暂无检测结果，请启动监控后开始检测
            </div>
        `;
        return;
    }

    // 生成卡片式详细结果（倒序显示，最新的在前）
    let resultsHtml = `<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px;">`;

    // 倒序遍历，最新的结果显示在前面
    for (let i = monitorResults.length - 1; i >= 0; i--) {
        const r = monitorResults[i];
        const uniqueId = `monitor-${i}`;
        const isAnomaly = r.is_anomaly;
        const isNoMatch = r.status === '未匹配';

        // 根据状态设置颜色
        let statusColor, statusBg, statusText;
        if (isNoMatch) {
            statusColor = '#856404';
            statusBg = '#fff3cd';
            statusText = '⚠️ 未匹配';
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
        const timestamp = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();

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
                        <div><strong>异常分数:</strong> ${r.anomaly_score ? r.anomaly_score.toFixed(4) : 'N/A'}</div>
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <strong>状态:</strong>
                            <span style="background: ${statusBg}; color: ${statusColor}; padding: 2px 6px; border-radius: 4px; font-size: 11px;">
                                ${statusText}
                            </span>
                        </div>
                        <div><strong>时间:</strong> ${timestamp}</div>
                        <div><strong>序号:</strong> #${monitorResults.length - i}</div>
                    </div>
                    ${hasHeatmap ? `
                        <div style="margin-top: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <div style="font-size: 11px; color: #666;">🔥 异常热力图:</div>
                                <div style="display: flex; gap: 4px;">
                                    <button id="btn-monitor-original-${uniqueId}"
                                            onclick="toggleMonitorHeatmapMode('${uniqueId}', 'original')"
                                            style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">原图</button>
                                    <button id="btn-monitor-overlay-${uniqueId}"
                                            onclick="toggleMonitorHeatmapMode('${uniqueId}', 'overlay')"
                                            style="padding: 2px 8px; font-size: 10px; border: 1px solid #667eea; background: #667eea; color: white; border-radius: 4px; cursor: pointer;">叠加</button>
                                    <button id="btn-monitor-heatmap-${uniqueId}"
                                            onclick="toggleMonitorHeatmapMode('${uniqueId}', 'heatmap')"
                                            style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">纯热力图</button>
                                </div>
                            </div>
                            <img id="monitor-heatmap-img-${uniqueId}"
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
    resultsDiv.innerHTML = resultsHtml;
}

// 实时监控热力图模式切换
function toggleMonitorHeatmapMode(uniqueId, mode) {
    const img = document.getElementById(`monitor-heatmap-img-${uniqueId}`);
    const btnOriginal = document.getElementById(`btn-monitor-original-${uniqueId}`);
    const btnOverlay = document.getElementById(`btn-monitor-overlay-${uniqueId}`);
    const btnHeatmap = document.getElementById(`btn-monitor-heatmap-${uniqueId}`);

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

function clearMonitorResults() {
    monitorResults = []; // 清除存储的结果
    document.getElementById('monitorResults').innerHTML = `
        <div style="text-align: center; color: #999; padding: 40px;">
            暂无检测结果，请启动监控后开始检测
        </div>
    `;
    monitorLog('结果已清空');
}
