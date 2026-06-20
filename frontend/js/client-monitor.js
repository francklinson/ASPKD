// ==================== 客户端监控模块 ====================

        let clientMonitorInterval = null;
        let clientResults = [];
        let clientSelectedReferences = [];
        let clientWs = null;

        // localStorage 键名
        const CLIENT_RESULTS_KEY = 'client_monitor_results';

        function initClientMonitor() {
            console.log('[ClientMonitor] 初始化客户端监控');

            // 从 localStorage 加载历史结果
            loadClientResultsFromStorage();

            // 初始化参考音频选择（设备列表和参考音频列表由页面统一的load函数加载）
            initClientReferenceSelect();

            // 初始化算法和设备选择监听（自动保存配置）
            initClientConfigListeners();

            // 页面加载时自动刷新客户端状态
            refreshClientStatus();

            // 连接客户端监控WebSocket
            connectClientWebSocket();
        }

        // 从 localStorage 加载历史结果
        function loadClientResultsFromStorage() {
            try {
                const stored = localStorage.getItem(CLIENT_RESULTS_KEY);
                if (stored) {
                    clientResults = JSON.parse(stored);
                    console.log(`[ClientMonitor] 从本地存储加载了 ${clientResults.length} 条历史结果`);

                    // 更新显示
                    updateClientResultsDisplay();
                    updateClientStats();

                    // 显示提示
                    if (clientResults.length > 0) {
                        clientLog(`📂 已恢复 ${clientResults.length} 条历史检测结果`);
                    }
                }
            } catch (error) {
                console.error('[ClientMonitor] 加载历史结果失败:', error);
                clientResults = [];
            }
        }

        // 保存结果到 localStorage
        function saveClientResultsToStorage() {
            try {
                localStorage.setItem(CLIENT_RESULTS_KEY, JSON.stringify(clientResults));
            } catch (error) {
                console.error('[ClientMonitor] 保存结果到本地存储失败:', error);
            }
        }

        // 清空本地存储的结果
        function clearClientResultsStorage() {
            try {
                localStorage.removeItem(CLIENT_RESULTS_KEY);
            } catch (error) {
                console.error('[ClientMonitor] 清空本地存储失败:', error);
            }
        }

        // 初始化配置监听器（算法、设备变化时自动保存）
        function initClientConfigListeners() {
            const algorithmSelect = document.getElementById('clientAlgorithm');
            const deviceSelect = document.getElementById('clientDevice');

            if (algorithmSelect) {
                algorithmSelect.addEventListener('change', () => {
                    clientLog(`🎵 算法已切换: ${algorithmSelect.value}`);
                    autoSaveClientConfig();
                });
            }

            if (deviceSelect) {
                deviceSelect.addEventListener('change', () => {
                    clientLog(`📍 设备已切换: ${deviceSelect.value}`);
                    autoSaveClientConfig();
                });
            }
        }

        // 自动保存客户端配置
        async function autoSaveClientConfig() {
            const selectedAlgorithm = document.getElementById('clientAlgorithm').value;
            const selectedDevice = document.getElementById('clientDevice').value;
            const selectedRefs = clientSelectedReferences.map(r => r.path);

            try {
                const response = await fetch(`${API_BASE}/api/client/config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        algorithm: selectedAlgorithm,
                        device: selectedDevice,
                        reference_audios: selectedRefs
                    })
                });

                if (response.ok) {
                    console.log('[ClientMonitor] 配置已自动保存');
                } else {
                    console.error('[ClientMonitor] 自动保存配置失败');
                }
            } catch (error) {
                console.error('[ClientMonitor] 自动保存配置错误:', error);
            }
        }

        // 初始化参考音频选择
        function initClientReferenceSelect() {
            const addBtn = document.getElementById('addClientRefBtn');
            const select = document.getElementById('clientReferenceAudioSelect');

            if (addBtn && select) {
                addBtn.addEventListener('click', () => {
                    const path = select.value;
                    const name = select.options[select.selectedIndex].text;

                    if (!path) {
                        showModal('请先选择一个参考音频', '提示');
                        return;
                    }

                    // 检查是否已添加
                    if (clientSelectedReferences.some(r => r.path === path)) {
                        showModal('该参考音频已添加', '提示');
                        return;
                    }

                    clientSelectedReferences.push({ path, name });
                    updateClientSelectedRefsDisplay();

                    // 清空选择
                    select.value = '';

                    clientLog(`已添加参考音频: ${name}`);

                    // 自动保存配置
                    autoSaveClientConfig();
                });
            }
        }

        // 更新参考音频显示
        function updateClientSelectedRefsDisplay() {
            const container = document.getElementById('clientSelectedRefs');
            const hiddenInput = document.getElementById('clientReferenceAudios');

            if (clientSelectedReferences.length === 0) {
                container.innerHTML = '<span style="color: #999; font-size: 12px; line-height: 20px;">未选择参考音频，将使用自动匹配</span>';
                hiddenInput.value = '';
            } else {
                container.innerHTML = clientSelectedReferences.map((ref, index) => `
                    <div style="display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; background: #e3f2fd; border: 1px solid #90caf9; border-radius: 16px; font-size: 12px; color: #1976d2;">
                        <span>${ref.name}</span>
                        <button type="button" onclick="removeClientReference(${index})" style="background: none; border: none; color: #1976d2; cursor: pointer; padding: 0; font-size: 14px; line-height: 1;">&times;</button>
                    </div>
                `).join('');

                // 更新隐藏字段
                hiddenInput.value = JSON.stringify(clientSelectedReferences.map(r => r.path));
            }
        }

        // 移除参考音频
        function removeClientReference(index) {
            const ref = clientSelectedReferences[index];
            clientSelectedReferences.splice(index, 1);
            updateClientSelectedRefsDisplay();
            clientLog(`已移除参考音频: ${ref.name}`);

            // 自动保存配置
            autoSaveClientConfig();
        }

        // 连接客户端监控WebSocket
        function connectClientWebSocket() {
            const wsUrl = `ws://${window.location.host}/ws/progress/broadcast`;
            clientWs = new WebSocket(wsUrl);

            clientWs.onopen = () => {
                console.log('[ClientMonitor] WebSocket connected');
                clientLog('WebSocket 已连接');
            };

            clientWs.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'client_upload') {
                    // 客户端上传新文件
                    clientLog(`📤 客户端上传: ${data.data?.filename || '未知文件'}`);
                } else if (data.type === 'monitor_update') {
                    // 客户端检测结果 - 使用与实时监控相同的格式
                    updateClientStats(data.data);
                } else if (data.type === 'monitor_log') {
                    // 监控日志
                    clientLog(data.data.message);
                }
            };

            clientWs.onerror = (error) => {
                console.error('[ClientMonitor] WebSocket error:', error);
                clientLog('WebSocket 错误');
            };

            clientWs.onclose = () => {
                console.log('[ClientMonitor] WebSocket closed');
                // 尝试重连
                setTimeout(connectClientWebSocket, 5000);
            };
        }

        // 更新客户端统计 - 与实时监控对齐
        function updateClientStats(data) {
            if (data) {
                if (data.total_processed !== undefined) {
                    document.getElementById('clientTotalFiles').textContent = data.total_processed;
                }
                if (data.anomaly_count !== undefined) {
                    document.getElementById('clientTotalAnomaly').textContent = data.anomaly_count;
                }

                if (data.latest_result) {
                    // 保存结果用于显示热力图
                    clientResults.push(data.latest_result);

                    // 保存到 localStorage
                    saveClientResultsToStorage();
                    // 更新卡片式展示
                    updateClientResultsDisplay();
                }
            } else {
                // 无参数调用时，只更新统计显示
                const totalFiles = clientResults.length;
                const anomalyCount = clientResults.filter(r => r.is_anomaly).length;

                document.getElementById('clientTotalFiles').textContent = totalFiles;
                document.getElementById('clientTotalAnomaly').textContent = anomalyCount;
            }
        }

        // 更新结果显示 - 与实时监控完全对齐
        function updateClientResultsDisplay() {
            const resultsDiv = document.getElementById('clientResults');

            if (clientResults.length === 0) {
                resultsDiv.innerHTML = `
                    <div style="text-align: center; color: #999; padding: 40px;">
                        等待客户端上传文件...
                    </div>
                `;
                return;
            }

            // 生成卡片式详细结果（倒序显示，最新的在前）- 与实时监控一致
            let resultsHtml = `<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px;">`;

            // 倒序遍历，最新的结果显示在前面
            for (let i = clientResults.length - 1; i >= 0; i--) {
                const r = clientResults[i];
                const uniqueId = `client-${i}`;
                const isAnomaly = r.is_anomaly;
                const isNoMatch = r.status === '未匹配';

                // 根据状态设置颜色 - 与实时监控一致
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
                        ${r.client_name ? `<div style="padding: 8px 12px; background: #f8f9fa; border-bottom: 1px solid #e0e0e0; font-size: 11px; color: #666;">客户端: ${r.client_name}</div>` : ''}
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
                                <div style="display: flex; align-items: center; gap: 6px;">
                                    <strong>参考音频:</strong> ${r.segment_info && r.segment_info.music_name ? r.segment_info.music_name : '未匹配'}
                                </div>
                                <div><strong>序号:</strong> #${clientResults.length - i}</div>
                            </div>
                            ${hasHeatmap ? `
                                <div style="margin-top: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <div style="font-size: 11px; color: #666;">🔥 异常热力图:</div>
                                        <div style="display: flex; gap: 4px;">
                                            <button id="btn-client-original-${uniqueId}"
                                                    onclick="toggleClientHeatmapMode('${uniqueId}', 'original')"
                                                    style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">原图</button>
                                            <button id="btn-client-overlay-${uniqueId}"
                                                    onclick="toggleClientHeatmapMode('${uniqueId}', 'overlay')"
                                                    style="padding: 2px 8px; font-size: 10px; border: 1px solid #667eea; background: #667eea; color: white; border-radius: 4px; cursor: pointer;">叠加</button>
                                            <button id="btn-client-heatmap-${uniqueId}"
                                                    onclick="toggleClientHeatmapMode('${uniqueId}', 'heatmap')"
                                                    style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">纯热力图</button>
                                        </div>
                                    </div>
                                    <img id="client-heatmap-img-${uniqueId}"
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

        // 刷新客户端状态
        async function refreshClientStatus() {
            try {
                // 先调用刷新状态接口，更新离线客户端
                const refreshResponse = await fetch(`${API_BASE}/api/client/refresh-status`, {
                    method: 'POST'
                });
                if (refreshResponse.ok) {
                    const refreshData = await refreshResponse.json();
                    if (refreshData.updated_count > 0) {
                        console.log(`[ClientMonitor] 已更新 ${refreshData.updated_count} 个客户端状态为离线`);
                    }
                }

                // 然后获取最新客户端列表
                const response = await fetch(`${API_BASE}/api/client/status`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                // 更新统计
                document.getElementById('clientTotalActive').textContent = data.total_active;

                // 计算总文件数
                const totalFiles = data.clients.reduce((sum, c) => sum + c.total_uploaded, 0);
                document.getElementById('clientTotalFiles').textContent = totalFiles;

                // 渲染客户端列表
                renderClientList(data.clients);

            } catch (error) {
                console.error('[ClientMonitor] 获取客户端状态失败:', error);
            }
        }

        // 渲染客户端列表
        function renderClientList(clients) {
            const container = document.getElementById('clientList');

            if (clients.length === 0) {
                container.innerHTML = `
                    <div style="text-align: center; color: #999; padding: 40px;">
                        <div style="font-size: 48px; margin-bottom: 16px;">📡</div>
                        <div>暂无客户端连接</div>
                        <div style="font-size: 12px; margin-top: 8px;">
                            请在客户端运行 client_monitor.py 脚本
                        </div>
                    </div>
                `;
                return;
            }

            // 使用网格布局，一行显示多个卡片
            container.style.display = 'grid';
            container.style.gridTemplateColumns = 'repeat(auto-fill, minmax(160px, 1fr))';
            container.style.gap = '8px';

            container.innerHTML = clients.map(client => {
                const statusColor = client.status === 'online' ? '#28a745' : '#dc3545';
                const statusIcon = client.status === 'online' ? '🟢' : '🔴';
                const lastHeartbeat = new Date(client.last_heartbeat);
                const timeAgo = getTimeAgo(lastHeartbeat);
                const connectedAt = new Date(client.connected_at);
                const connectionDuration = getDurationText(connectedAt);

                // 计算异常率（上限100%）
                const anomalyRate = client.total_uploaded > 0
                    ? Math.min((client.anomaly_detected / client.total_uploaded) * 100, 100).toFixed(1)
                    : '0.0';

                return `
                    <div style="border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px; background: ${client.status === 'online' ? '#f8fff8' : '#fff8f8'}; font-size: 10px; min-width: 0;">
                        <!-- 第一行：名称+状态 -->
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <div style="font-weight: 600; font-size: 11px; display: flex; align-items: center; gap: 2px; overflow: hidden;">
                                ${statusIcon}
                                <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${client.name}</span>
                            </div>
                            <div style="font-size: 8px; color: ${statusColor}; background: ${statusColor}20; padding: 0px 3px; border-radius: 3px; flex-shrink: 0; margin-left: 4px;">
                                ${client.status === 'online' ? '在线' : '离线'}
                            </div>
                        </div>

                        <!-- 第二行：IP -->
                        <div style="font-size: 9px; color: #888; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            🌐 ${client.ip_address || '未知IP'}
                        </div>

                        <!-- 第三行：时长 + 最后活动 -->
                        <div style="font-size: 9px; color: #888; margin-bottom: 4px; display: flex; justify-content: space-between;">
                            <span>⏱️ ${connectionDuration}</span>
                            <span style="color: #aaa;">💓 ${timeAgo}</span>
                        </div>

                        <!-- 第四行：统计数据 -->
                        <div style="font-size: 9px; color: #666; display: flex; justify-content: space-between; padding-top: 4px; border-top: 1px solid #eee;">
                            <span title="上传文件数">📤 ${client.total_uploaded}</span>
                            <span title="异常数" style="color: ${client.anomaly_detected > 0 ? '#dc3545' : '#666'};">⚠️ ${client.anomaly_detected}</span>
                            <span title="异常率" style="color: ${anomalyRate > 0 ? '#dc3545' : '#28a745'};">📊 ${anomalyRate}%</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // 获取连接时长文本
        function getDurationText(startDate) {
            const now = new Date();
            const diff = Math.floor((now - startDate) / 1000);

            if (diff < 60) return `${diff}秒`;
            if (diff < 3600) return `${Math.floor(diff / 60)}分钟`;
            if (diff < 86400) return `${Math.floor(diff / 3600)}小时`;
            const days = Math.floor(diff / 86400);
            const hours = Math.floor((diff % 86400) / 3600);
            return hours > 0 ? `${days}天${hours}小时` : `${days}天`;
        }

        // 获取相对时间
        function getTimeAgo(date) {
            const now = new Date();
            const diff = Math.floor((now - date) / 1000);

            if (diff < 60) return `${diff}秒前`;
            if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`;
            if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`;
            return `${Math.floor(diff / 86400)}天前`;
        }

        // 客户端日志
        function clientLog(message) {
            const logBox = document.getElementById('clientLog');
            if (!logBox) return;

            const time = new Date().toLocaleTimeString();
            const line = document.createElement('div');
            line.style.cssText = 'padding: 4px 0; border-bottom: 1px solid #f0f0f0; font-size: 12px;';
            line.innerHTML = `<span style="color: #999;">[${time}]</span> ${message}`;
            logBox.appendChild(line);
            logBox.scrollTop = logBox.scrollHeight;

            // 限制日志行数
            while (logBox.children.length > 100) {
                logBox.removeChild(logBox.firstChild);
            }
        }

        // 定期刷新客户端状态（每10秒）
        setInterval(refreshClientStatus, 10000);
