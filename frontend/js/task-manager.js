// ==================== 任务管理模块（支持离线 + 在线检测） ====================

// 分页和筛选状态
let currentPage = 1;
const PAGE_SIZE = 10;
let totalTasks = 0;
let totalPages = 1;
let currentStatusFilter = '';
let currentTypeFilter = 'all'; // 'all' | 'offline' | 'online'
let cachedTasks = [];  // 当前页数据缓存（含在线+离线）

// ==================== 统计信息 ====================

async function loadTaskStats() {
    try {
        // 并行获取离线任务统计和在线检测统计
        const [offlineRes, onlineRes] = await Promise.allSettled([
            fetch(`${API_BASE}/api/tasks/stats`),
            fetch(`${API_BASE}/api/client/stats`)
        ]);

        let total = 0, running = 0, queued = 0, completed = 0, failed = 0;

        if (offlineRes.status === 'fulfilled' && offlineRes.value.ok) {
            const data = await offlineRes.value.json();
            total += data.total || 0;
            running = data.running || 0;
            queued = data.queued || 0;
            completed += data.completed || 0;
            failed = data.failed || 0;
        }

        if (onlineRes.status === 'fulfilled' && onlineRes.value.ok) {
            const data = await onlineRes.value.json();
            total += data.total || 0;
            completed += data.total || 0;  // 在线结果始终视为"已完成"
        }

        document.getElementById('statTotal').textContent = total;
        document.getElementById('statRunning').textContent = running;
        document.getElementById('statQueued').textContent = queued;
        document.getElementById('statCompleted').textContent = completed;
        document.getElementById('statFailed').textContent = failed;
    } catch (error) {
        console.error('加载任务统计失败:', error);
    }
}

async function loadOnlineStats() {
    try {
        const response = await fetch(`${API_BASE}/api/client/stats`);
        if (!response.ok) return { total: 0 };
        return await response.json();
    } catch {
        return { total: 0 };
    }
}

// ==================== 在线结果规范化 ====================

function normalizeOnlineTask(r) {
    /** 将在线检测结果 dict 标准化为类 Task 格式以便统一渲染 */
    const timestamp = r.timestamp || new Date().toISOString();
    const displayId = r.result_id ? `online_${r.result_id}` : `online_${r.filename}_${timestamp}`;
    return {
        id: displayId,
        _type: 'online',
        _result_id: r.result_id,
        _client_name: r.client_name || '-',
        _filename: r.filename || '-',
        _anomaly_score: r.anomaly_score,
        _is_anomaly: r.is_anomaly,
        _status_detail: r.status || '正常',
        _original_path: r.original_path,
        _overlay_path: r.overlay_path,
        _heatmap_path: r.heatmap_path,
        _audio_slice_path: r.audio_slice_path,
        _segment_info: r.segment_info,
        _filepath: r.filepath,
        _raw_result: r,
        // 映射为标准任务字段
        status: 'completed',
        algorithm: r.algorithm || '—',
        file_count: 1,
        progress: 100,
        created_at: timestamp,
        started_at: timestamp,
        completed_at: timestamp,
    };
}

// ==================== 任务列表 ====================

async function loadTaskList() {
    try {
        // 加载统计
        await loadTaskStats();

        const typeFilter = document.getElementById('filterType')?.value || currentTypeFilter;
        currentTypeFilter = typeFilter;

        let allItems = [];

        // 根据类型筛选，加载对应数据
        if (typeFilter === 'all' || typeFilter === 'offline') {
            // 全部模式下多取一些离线任务，便于与在线结果合并排序
            const offlineLimit = typeFilter === 'all' ? 100 : PAGE_SIZE;
            const offlineItems = await loadOfflineTasks(offlineLimit);
            allItems.push(...offlineItems);
        }

        if (typeFilter === 'all' || typeFilter === 'online') {
            const onlineItems = await loadOnlineResults();
            allItems.push(...onlineItems);
        }

        // 按 created_at 降序排序（最新在前）
        allItems.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

        totalTasks = allItems.length;
        totalPages = Math.max(1, Math.ceil(totalTasks / PAGE_SIZE));

        // 前端搜索过滤
        const searchText = document.getElementById('filterSearch').value.trim().toLowerCase();
        if (searchText) {
            allItems = allItems.filter(t => t.id.toLowerCase().includes(searchText));
        }

        // 算法过滤
        const algoFilter = document.getElementById('filterAlgorithm').value;
        if (algoFilter) {
            allItems = allItems.filter(t => t.algorithm === algoFilter);
        }

        // 状态过滤（仅对离线任务生效）
        const statusFilter = document.getElementById('filterStatus').value || currentStatusFilter;
        if (statusFilter) {
            allItems = allItems.filter(t => t._type !== 'online' && t.status === statusFilter);
        }

        // 重新计算分页（过滤后）
        totalTasks = allItems.length;
        totalPages = Math.max(1, Math.ceil(totalTasks / PAGE_SIZE));
        if (currentPage > totalPages) currentPage = totalPages;

        // 取当前页数据
        const startIdx = (currentPage - 1) * PAGE_SIZE;
        const pageItems = allItems.slice(startIdx, startIdx + PAGE_SIZE);
        cachedTasks = pageItems;

        // 更新任务计数信息
        document.getElementById('taskCountInfo').textContent =
            `共 ${totalTasks} 个任务/结果，当前第 ${currentPage}/${totalPages} 页`;

        const tbody = document.querySelector('#taskList tbody');

        if (pageItems.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: #999; padding: 40px;">暂无任务</td></tr>';
            document.getElementById('taskPagination').style.display = 'none';
            return;
        }

        // 填充算法筛选下拉（从所有数据中提取）
        populateAlgorithmFilter(allItems);

        tbody.innerHTML = pageItems.map(t => {
            if (t._type === 'online') {
                return renderOnlineTaskRow(t);
            } else {
                return renderOfflineTaskRow(t);
            }
        }).join('');

        renderPagination();
    } catch (error) {
        console.error('加载任务列表失败:', error);
        const tbody = document.querySelector('#taskList tbody');
        tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: #ff4d4f; padding: 40px;">加载失败: ${error.message}</td></tr>`;
    }
}

async function loadOfflineTasks(limitOverride) {
    /** 加载离线检测任务（分页） */
    try {
        const params = new URLSearchParams();
        const statusFilter = document.getElementById('filterStatus').value || currentStatusFilter;
        if (statusFilter) params.set('status', statusFilter);
        // 支持外部指定 limit（全部模式下取更多），否则使用当前页大小
        const actualLimit = limitOverride || PAGE_SIZE;
        params.set('limit', String(actualLimit));
        params.set('offset', String(limitOverride ? 0 : (currentPage - 1) * PAGE_SIZE));

        const response = await fetch(`${API_BASE}/api/tasks/list?${params}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();

        // 给离线任务添加 _type 标记
        return (data.tasks || []).map(t => ({ ...t, _type: 'offline' }));
    } catch (error) {
        console.error('加载离线任务失败:', error);
        return [];
    }
}

async function loadOnlineResults() {
    /** 加载在线检测结果 */
    try {
        // 在线结果数量有限（max 1000），一次性加载全部
        const response = await fetch(`${API_BASE}/api/client/results?limit=1000&offset=0`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();

        return (data.results || []).map(r => normalizeOnlineTask(r));
    } catch (error) {
        console.error('加载在线检测结果失败:', error);
        return [];
    }
}

// ==================== 表格行渲染 ====================

function renderOfflineTaskRow(t) {
    /** 渲染离线检测任务行 */
    const statusBadge = getStatusBadge(t.status);
    const duration = getTaskDuration(t);
    const progressColor = t.progress >= 100 ? '#52c41a' : '#667eea';

    return `
        <tr>
            <td><span class="badge badge-secondary" style="background:#e8e8e8;color:#666;">离线</span></td>
            <td><code title="${t.id}" style="font-size:12px;cursor:pointer;" onclick="showTaskDetailModal('${t.id}')">${t.id.substring(0, 8)}...</code></td>
            <td><span class="badge badge-${getStatusColor(t.status)}">${statusBadge}</span></td>
            <td style="font-size:12px;color:#999;">—</td>
            <td>${t.algorithm}</td>
            <td>${t.file_count}</td>
            <td>
                <div class="progress-mini">
                    <div class="progress-mini-bar">
                        <div class="progress-mini-fill" style="width:${Math.min(t.progress, 100)}%;background:${progressColor};"></div>
                    </div>
                    <span style="font-size:12px;color:#666;min-width:38px;">${t.progress.toFixed(0)}%</span>
                </div>
            </td>
            <td style="font-size:12px;white-space:nowrap;">${formatTime(t.created_at)}</td>
            <td style="font-size:12px;color:#999;">${duration}</td>
            <td style="white-space:nowrap;">
                <button class="btn btn-secondary" onclick="showTaskDetailModal('${t.id}')" style="padding:3px 8px;font-size:11px;">详情</button>
                ${t.status === 'running' ? `<button class="btn btn-danger" onclick="cancelTask('${t.id}')" style="padding:3px 8px;font-size:11px;margin-left:3px;">取消</button>` : ''}
                ${t.status === 'failed' ? `<button class="btn btn-primary" onclick="retryTask('${t.id}')" style="padding:3px 8px;font-size:11px;margin-left:3px;">重试</button>` : ''}
                ${['completed', 'failed', 'cancelled'].includes(t.status) ? `<button class="btn btn-danger" onclick="deleteSingleTask('${t.id}', 'offline')" style="padding:3px 8px;font-size:11px;margin-left:3px;background:#999;color:white;">删除</button>` : ''}
            </td>
        </tr>
    `;
}

function renderOnlineTaskRow(t) {
    /** 渲染在线检测结果行 */
    const anomalyScore = t._anomaly_score !== undefined ? t._anomaly_score.toFixed(4) : '-';
    const statusIcon = t._is_anomaly ? '🔴' : (t._status_detail === '未匹配' ? '⚠️' : '🟢');
    const statusColor = t._is_anomaly ? '#dc3545' : (t._status_detail === '未匹配' ? '#ffc107' : '#52c41a');

    return `
        <tr>
            <td><span class="badge" style="background:#e6f7ff;color:#1890ff;border:1px solid #91d5ff;">在线</span></td>
            <td><code style="font-size:12px;cursor:pointer;" onclick="showOnlineDetailModal('${t._result_id}')">${t.id.substring(0, 12)}...</code></td>
            <td><span class="badge badge-success">✅ 已完成</span></td>
            <td style="font-size:12px;">${t._client_name}</td>
            <td>${t.algorithm}</td>
            <td>1</td>
            <td style="font-size:12px;color:#999;">—</td>
            <td style="font-size:12px;white-space:nowrap;">${formatTime(t.created_at)}</td>
            <td style="font-size:12px;color:#999;">—</td>
            <td style="white-space:nowrap;">
                <button class="btn btn-secondary" onclick="showOnlineDetailModal('${t._result_id}')" style="padding:3px 8px;font-size:11px;">详情</button>
                <button class="btn btn-danger" onclick="deleteSingleTask('${t._result_id}', 'online')" style="padding:3px 8px;font-size:11px;margin-left:3px;background:#999;color:white;">删除</button>
            </td>
        </tr>
    `;
}

// ==================== 筛选与分页 ====================

function filterByStatus(status) {
    currentStatusFilter = status;
    currentPage = 1;

    // 更新统计卡片的选中样式
    document.querySelectorAll('#taskStats .stat-card').forEach(card => {
        card.classList.toggle('active-filter', card.dataset.filter === status);
    });

    // 同步下拉框
    document.getElementById('filterStatus').value = status;

    loadTaskList();
}

function applyFilters() {
    currentPage = 1;
    currentStatusFilter = '';
    currentTypeFilter = document.getElementById('filterType')?.value || 'all';

    // 清除统计卡片的选中样式
    document.querySelectorAll('#taskStats .stat-card').forEach(card => {
        card.classList.remove('active-filter');
    });

    loadTaskList();
}

function goToPage(page) {
    if (page < 1 || page > totalPages) return;
    currentPage = page;
    loadTaskList();
}

function renderPagination() {
    const pagination = document.getElementById('taskPagination');
    if (totalPages <= 1) {
        pagination.style.display = 'none';
        return;
    }

    pagination.style.display = 'flex';
    document.getElementById('pageFirst').disabled = currentPage <= 1;
    document.getElementById('pagePrev').disabled = currentPage <= 1;
    document.getElementById('pageNext').disabled = currentPage >= totalPages;
    document.getElementById('pageLast').disabled = currentPage >= totalPages;
    document.getElementById('pageInfo').textContent = `第 ${currentPage}/${totalPages} 页`;
}

// 填充算法筛选下拉
function populateAlgorithmFilter(tasks) {
    const select = document.getElementById('filterAlgorithm');
    // 只在初始化时填充
    if (select.options.length > 1) return;

    const algorithms = [...new Set(tasks.map(t => t.algorithm).filter(Boolean))];
    algorithms.forEach(algo => {
        const opt = document.createElement('option');
        opt.value = algo;
        opt.textContent = algo;
        select.appendChild(opt);
    });
}

// ==================== 任务状态工具 ====================

function getStatusColor(status) {
    const map = {
        'pending': 'secondary',
        'running': 'info',
        'completed': 'success',
        'failed': 'danger',
        'cancelled': 'secondary'
    };
    return map[status] || 'secondary';
}

function getStatusBadge(status) {
    const map = {
        'pending': '⏳ 排队中',
        'running': '⚡ 运行中',
        'completed': '✅ 已完成',
        'failed': '❌ 失败',
        'cancelled': '🗑️ 已取消'
    };
    return map[status] || status;
}

function getTaskDuration(task) {
    if (task._type === 'online') return '—';
    if (task.status === 'pending' || task.status === 'running') {
        if (task.started_at) {
            return `进行中 ${getTimeElapsed(task.started_at)}`;
        }
        return '等待中';
    }
    if (task.completed_at && task.started_at) {
        return getTimeDiff(task.started_at, task.completed_at);
    }
    return '-';
}

function getTimeElapsed(timeStr) {
    const start = new Date(timeStr).getTime();
    const now = Date.now();
    const diff = Math.floor((now - start) / 1000);
    return formatDuration(diff);
}

function getTimeDiff(startStr, endStr) {
    const start = new Date(startStr).getTime();
    const end = new Date(endStr).getTime();
    const diff = Math.floor((end - start) / 1000);
    return formatDuration(diff);
}

function formatDuration(seconds) {
    if (seconds < 60) return `${seconds}秒`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分${seconds % 60}秒`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}时${m}分`;
}

function formatTime(timeStr) {
    if (!timeStr) return '-';
    try {
        return new Date(timeStr).toLocaleString('zh-CN', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return timeStr;
    }
}

// ==================== 任务详情（离线） ====================

async function showTaskDetailModal(taskId) {
    try {
        const response = await fetch(`${API_BASE}/api/detection/result/${taskId}`);
        if (!response.ok) {
            throw new Error('获取任务详情失败');
        }
        const data = await response.json();

        renderTaskDetailHtml(taskId, data);
    } catch (error) {
        console.error('查看任务失败:', error);
        await showModal('查看任务详情失败: ' + error.message, '错误');
    }
}

function renderTaskDetailHtml(taskId, data) {
    const statusBadge = getStatusBadge(data.status);
    const statusColor = data.status === 'completed' ? '#52c41a' :
                        data.status === 'failed' ? '#ff4d4f' :
                        data.status === 'running' ? '#1890ff' : '#999';

    const progressColor = data.progress >= 100 ? '#52c41a' : '#667eea';

    // 构建结构化详情HTML
    let html = `
        <div class="task-detail-section">
            <h4>📋 基本信息</h4>
            <div class="task-detail-grid">
                <div class="task-detail-item"><span class="label">类型</span><span class="value"><span class="badge badge-secondary" style="background:#e8e8e8;color:#666;">离线任务</span></span></div>
                <div class="task-detail-item"><span class="label">任务ID</span><span class="value"><code>${taskId.substring(0, 12)}...</code></span></div>
                <div class="task-detail-item"><span class="label">状态</span><span class="value" style="color:${statusColor}">${statusBadge}</span></div>
                <div class="task-detail-item"><span class="label">算法</span><span class="value">${data.algorithm || '-'}</span></div>
                <div class="task-detail-item"><span class="label">设备</span><span class="value">${data.device || '-'}</span></div>
                <div class="task-detail-item"><span class="label">进度</span><span class="value">${data.progress?.toFixed(1) || 0}%</span></div>
                <div class="task-detail-item"><span class="label">文件数</span><span class="value">${data.results?.length || 0} 个</span></div>
                <div class="task-detail-item" style="grid-column:1/-1;">
                    <div class="progress-mini" style="width:100%;">
                        <div class="progress-mini-bar" style="flex:1;">
                            <div class="progress-mini-fill" style="width:${Math.min(data.progress || 0, 100)}%;background:${progressColor};"></div>
                        </div>
                        <span style="font-size:12px;color:#666;">${(data.progress || 0).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    if (data.current_file) {
        html += `
            <div class="task-detail-section">
                <h4>📄 当前文件</h4>
                <div style="font-size:13px;padding:8px 12px;background:#f8f9fa;border-radius:4px;">${data.current_file}</div>
            </div>
        `;
    }

    if (data.error) {
        html += `
            <div class="task-detail-section">
                <h4>❌ 错误信息</h4>
                <div style="font-size:13px;padding:8px 12px;background:#fff2f0;border:1px solid #ffccc7;border-radius:4px;color:#cf1322;">${data.error}</div>
            </div>
        `;
    }

    if (data.results && data.results.length > 0) {
        const anomalyCount = data.results.filter(r => r.is_anomaly).length;
        const failedCount = data.results.filter(r => r.status === '预处理失败').length;

        html += `
            <div class="task-detail-section">
                <h4>📊 检测结果 <span style="font-size:12px;font-weight:400;color:#999;">${data.results.length} 个文件，${anomalyCount} 个异常，${failedCount} 个失败</span></h4>
                <div class="task-detail-results">
        `;

        data.results.forEach((r, i) => {
            const isAnomaly = r.is_anomaly;
            const isFailed = r.status === '预处理失败';
            const resultColor = isFailed ? '#ffc107' : (isAnomaly ? '#dc3545' : '#52c41a');
            const resultIcon = isFailed ? '⚠️' : (isAnomaly ? '🔴' : '🟢');
            const scoreText = r.anomaly_score !== undefined ? r.anomaly_score.toFixed(4) : '-';

            html += `
                <div class="task-detail-result-item">
                    <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px;" title="${r.filename}">${r.filename}</span>
                    <span style="color:${resultColor};font-weight:500;white-space:nowrap;margin-left:12px;">
                        ${resultIcon} ${isFailed ? '失败' : (isAnomaly ? '异常' : '正常')}
                        <span style="color:#999;font-weight:400;margin-left:8px;">分数: ${scoreText}</span>
                    </span>
                </div>
            `;
        });

        html += `</div></div>`;
    }

    // 显示模态框
    const modal = document.getElementById('customModal');
    document.getElementById('modalTitle').textContent = '📋 任务详情';
    document.getElementById('modalMessage').innerHTML = html;
    document.getElementById('modalMessage').style.whiteSpace = 'normal';
    document.getElementById('modalFooter').innerHTML = `
        <button class="modal-btn modal-btn-primary" onclick="closeModal()">关闭</button>
    `;
    modal.classList.add('active');
}

// ==================== 任务详情（在线检测结果） ====================

async function showOnlineDetailModal(resultId) {
    try {
        // 从 /api/client/results 获取详情
        // 先获取全部结果找到对应的这条（没有单条查询接口）
        const response = await fetch(`${API_BASE}/api/client/results?limit=1000&offset=0`);
        if (!response.ok) throw new Error('获取在线检测结果失败');
        const data = await response.json();
        const result = (data.results || []).find(r => r.result_id == resultId);
        if (!result) throw new Error('结果不存在');

        renderOnlineDetailHtml(result);
    } catch (error) {
        console.error('查看在线结果详情失败:', error);
        await showModal('查看详情失败: ' + error.message, '错误');
    }
}

function renderOnlineDetailHtml(result) {
    const isAnomaly = result.is_anomaly;
    const isUnmatched = result.status === '未匹配';
    const statusColor = isAnomaly ? '#dc3545' : (isUnmatched ? '#ffc107' : '#52c41a');
    const statusIcon = isAnomaly ? '🔴' : (isUnmatched ? '⚠️' : '🟢');
    const statusText = result.status || (isAnomaly ? '异常' : '正常');
    const scoreText = result.anomaly_score !== undefined ? result.anomaly_score.toFixed(4) : '-';
    const clientName = result.client_name || '-';
    const algorithm = result.algorithm || '-';

    let html = `
        <div class="task-detail-section">
            <h4>📋 基本信息</h4>
            <div class="task-detail-grid">
                <div class="task-detail-item"><span class="label">类型</span><span class="value"><span class="badge" style="background:#e6f7ff;color:#1890ff;border:1px solid #91d5ff;">在线检测</span></span></div>
                <div class="task-detail-item"><span class="label">结果ID</span><span class="value"><code>${result.result_id}</code></span></div>
                <div class="task-detail-item"><span class="label">客户端</span><span class="value">${clientName}</span></div>
                <div class="task-detail-item"><span class="label">算法</span><span class="value">${algorithm}</span></div>
                <div class="task-detail-item"><span class="label">文件名</span><span class="value" style="word-break:break-all;">${result.filename || '-'}</span></div>
                <div class="task-detail-item"><span class="label">时间</span><span class="value">${formatTime(result.timestamp)}</span></div>
            </div>
        </div>
        <div class="task-detail-section">
            <h4>🎯 检测结果</h4>
            <div style="padding:12px 16px;background:#f8f9fa;border-radius:6px;">
                <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                    <span style="font-size:16px;font-weight:600;color:${statusColor};">${statusIcon} ${statusText}</span>
                    <span style="font-size:13px;color:#666;">异常分数: <strong>${scoreText}</strong></span>
                </div>
            </div>
        </div>
    `;

    // 参考音频匹配信息
    const segInfo = result.segment_info;
    if (segInfo && segInfo.music_name) {
        html += `
            <div class="task-detail-section">
                <h4>🎵 参考音频匹配</h4>
                <div class="task-detail-grid">
                    <div class="task-detail-item"><span class="label">匹配音频</span><span class="value">${segInfo.music_name}</span></div>
                    <div class="task-detail-item"><span class="label">开始时间</span><span class="value">${segInfo.start_time?.toFixed(1) || '-'}s</span></div>
                    <div class="task-detail-item"><span class="label">结束时间</span><span class="value">${segInfo.end_time?.toFixed(1) || '-'}s</span></div>
                    <div class="task-detail-item"><span class="label">置信度</span><span class="value">${segInfo.confidence?.toFixed(2) || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">匹配率</span><span class="value">${(segInfo.match_ratio * 100)?.toFixed(1) || '-'}%</span></div>
                </div>
            </div>
        `;
    }

    // 频谱图（如果存在）
    const overlayPath = result.overlay_path;
    if (overlayPath) {
        const imgUrl = `${API_BASE}/${overlayPath}`;
        html += `
            <div class="task-detail-section">
                <h4>📊 频谱图</h4>
                <div style="text-align:center;margin-top:8px;">
                    <img src="${imgUrl}" alt="频谱图" style="max-width:100%;max-height:300px;border-radius:6px;border:1px solid #eee;" onerror="this.style.display='none'">
                </div>
            </div>
        `;
    }

    // 异常结果附加信息
    if (result.original_path) {
        html += `<div style="font-size:11px;color:#999;margin-top:8px;padding:8px;background:#fafafa;border-radius:4px;">
            原始图谱: ${result.original_path}<br>
            热力图: ${result.heatmap_path || '-'}<br>
            音频切片: ${result.audio_slice_path || '-'}
        </div>`;
    }

    // 显示模态框
    const modal = document.getElementById('customModal');
    document.getElementById('modalTitle').textContent = '🌐 在线检测详情';
    document.getElementById('modalMessage').innerHTML = html;
    document.getElementById('modalMessage').style.whiteSpace = 'normal';
    document.getElementById('modalFooter').innerHTML = `
        <button class="modal-btn modal-btn-primary" onclick="closeModal()">关闭</button>
    `;
    modal.classList.add('active');
}

// ==================== 操作函数 ====================

async function cancelTask(taskId) {
    const confirmed = await showModal('确定要取消该任务吗？', '确认', 'confirm');
    if (!confirmed) return;

    try {
        const response = await fetch(`${API_BASE}/api/detection/cancel/${taskId}`, {
            method: 'POST'
        });
        if (!response.ok) {
            throw new Error('取消任务失败');
        }
        await showModal('任务已取消', '提示');
        loadTaskList();
    } catch (error) {
        console.error('取消任务失败:', error);
        await showModal('取消任务失败: ' + error.message, '错误');
    }
}

async function deleteSingleTask(taskIdOrResultId, type) {
    if (type === 'online') {
        // 删除在线检测结果
        const confirmed = await showModal('确定要删除该在线检测结果吗？', '确认删除', 'danger');
        if (!confirmed) return;

        try {
            const response = await fetch(`${API_BASE}/api/client/results/${taskIdOrResultId}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                throw new Error('删除结果失败');
            }
            await showModal('在线检测结果已删除', '提示');
            loadTaskList();
        } catch (error) {
            console.error('删除在线结果失败:', error);
            await showModal('删除在线结果失败: ' + error.message, '错误');
        }
    } else {
        // 删除离线任务记录
        const confirmed = await showModal('确定要删除该任务记录吗？\n（不会删除关联的文件）', '确认删除', 'danger');
        if (!confirmed) return;

        try {
            const response = await fetch(`${API_BASE}/api/tasks/${taskIdOrResultId}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                throw new Error('删除任务失败');
            }
            await showModal('任务已删除', '提示');
            loadTaskList();
        } catch (error) {
            console.error('删除任务失败:', error);
            await showModal('删除任务失败: ' + error.message, '错误');
        }
    }
}

async function retryTask(taskId) {
    // 重试实际上就是重新提交相同的文件进行检测
    // 目前可以通过引导用户到离线检测页重新上传来实现
    await showModal('重试功能：请切换到"💻 离线检测"页面，重新上传文件进行检测。\n\n当前暂不支持一键重试失败任务。', '提示');
}

async function cleanupOldTasks() {
    const confirmed = await showModal(
        '确定要清理所有已完成/失败/取消的离线任务记录及相关文件吗？\n\n这将删除：\n• 已完成/失败/取消的任务记录\n• 上传的音频文件 (uploads/)\n• 临时切片文件 (output/slices/)\n• 生成的热力图 (output/vis/)\n• 导出的Excel/Zip文件 (output/exports/)\n\n⚠️ 此操作不可恢复！\n\n（注：在线检测结果不受影响）',
        '确认清理',
        'danger'
    );
    if (!confirmed) return;

    try {
        const response = await fetch(`${API_BASE}/api/tasks/cleanup?clear_all=true&include_files=true`, {
            method: 'POST'
        });
        if (!response.ok) {
            throw new Error('清理任务失败');
        }
        const result = await response.json();
        const stats = result.file_stats || {};
        const message = `离线任务清理完成！\n\n任务记录: ${result.removed_count || 0} 个\n上传文件: ${stats.uploads || 0} 个\n临时切片: ${stats.slice || 0} 个\n热力图: ${stats.visualize || 0} 个\n导出文件: ${stats.exports || 0} 个`;
        await showModal(message, '清理完成');
        loadTaskList();
    } catch (error) {
        console.error('清理任务失败:', error);
        await showModal('清理任务失败: ' + error.message, '错误');
    }
}
