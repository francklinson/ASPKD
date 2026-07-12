// ==================== 任务管理模块（支持全部任务类型） ====================

// 分页和筛选状态
let currentPage = 1;
const PAGE_SIZE = 10;
let totalTasks = 0;
let totalPages = 1;
let currentStatusFilter = '';
let currentTypeFilter = 'all';
let cachedTasks = [];

// ==================== 统计信息 ====================

async function loadTaskStats() {
    try {
        const resp = await fetch(`${API_BASE}/api/tasks/stats/all`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        document.getElementById('statTotal').textContent = data.total || 0;
        document.getElementById('statRunning').textContent = data.running || 0;
        document.getElementById('statQueued').textContent = data.pending || 0;
        document.getElementById('statCompleted').textContent = data.completed || 0;
        document.getElementById('statFailed').textContent = data.failed || 0;
    } catch (error) {
        console.error('加载任务统计失败:', error);
    }
}

// ==================== 任务列表 ====================

async function loadTaskList() {
    try {
        await loadTaskStats();

        currentTypeFilter = document.getElementById('filterType')?.value || 'all';

        const params = new URLSearchParams();
        if (currentTypeFilter !== 'all') params.set('task_type', currentTypeFilter);

        const statusFilter = document.getElementById('filterStatus')?.value || currentStatusFilter;
        if (statusFilter) params.set('status', statusFilter);

        params.set('limit', '200');
        params.set('offset', '0');

        const response = await fetch(`${API_BASE}/api/tasks/all?${params}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        let allItems = data.tasks || [];

        // 前端搜索过滤
        const searchText = document.getElementById('filterSearch')?.value?.trim()?.toLowerCase() || '';
        if (searchText) {
            allItems = allItems.filter(t =>
                (t.id || '').toLowerCase().includes(searchText) ||
                (t.algorithm || '').toLowerCase().includes(searchText) ||
                (t.algorithm_family || '').toLowerCase().includes(searchText) ||
                (t.category || '').toLowerCase().includes(searchText)
            );
        }

        // 算法过滤
        const algoFilter = document.getElementById('filterAlgorithm')?.value;
        if (algoFilter) {
            allItems = allItems.filter(t => t.algorithm === algoFilter || t.algorithm_family === algoFilter);
        }

        totalTasks = allItems.length;
        totalPages = Math.max(1, Math.ceil(totalTasks / PAGE_SIZE));
        if (currentPage > totalPages) currentPage = totalPages;

        const startIdx = (currentPage - 1) * PAGE_SIZE;
        const pageItems = allItems.slice(startIdx, startIdx + PAGE_SIZE);
        cachedTasks = pageItems;

        document.getElementById('taskCountInfo').textContent =
            `共 ${totalTasks} 个任务，当前第 ${currentPage}/${totalPages} 页`;

        populateAlgorithmFilter(allItems);

        const tbody = document.querySelector('#taskList tbody');

        if (pageItems.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: #999; padding: 40px;">暂无任务</td></tr>';
            document.getElementById('taskPagination').style.display = 'none';
            return;
        }

        tbody.innerHTML = pageItems.map(t => renderTaskRow(t)).join('');
        renderPagination();
    } catch (error) {
        console.error('加载任务列表失败:', error);
        const tbody = document.querySelector('#taskList tbody');
        tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: #ff4d4f; padding: 40px;">加载失败: ${error.message}</td></tr>`;
    }
}

// ==================== 行渲染 ====================

function renderTaskRow(t) {
    const typeBadge = getTypeBadge(t.type);
    const statusBadge = getStatusBadge(t.status);
    const duration = getTaskDuration(t);
    const progressColor = (t.progress || 0) >= 100 ? '#52c41a' : '#667eea';

    const algoDisplay = t.algorithm_family
        ? `<span style="font-size:11px;color:#888;">[${_familyLabel(t.algorithm_family)}]</span> ${t.algorithm || '-'}`
        : (t.algorithm || '-');

    const extraInfo = t.type === 'training'
        ? `<span style="font-size:11px;color:#888;">${t.category || ''}</span>`
        : t.type === 'online'
        ? `<span style="font-size:11px;color:#888;">${t.client_name || ''}</span>`
        : '';

    return `
        <tr>
            <td>${typeBadge}</td>
            <td><code title="${t.id}" style="font-size:12px;cursor:pointer;" onclick="showUnifiedTaskDetail('${t.id}','${t.type}')">${t.id.substring(0, 8)}...</code></td>
            <td><span class="badge badge-${getStatusColor(t.status)}">${statusBadge}</span></td>
            <td style="font-size:12px;">${extraInfo}</td>
            <td>${algoDisplay}</td>
            <td>${t.file_count || (t.type === 'online' ? 1 : '-')}</td>
            <td>
                <div class="progress-mini">
                    <div class="progress-mini-bar">
                        <div class="progress-mini-fill" style="width:${Math.min(t.progress || 0, 100)}%;background:${progressColor};"></div>
                    </div>
                    <span style="font-size:12px;color:#666;min-width:38px;">${(t.progress || 0).toFixed(0)}%</span>
                </div>
            </td>
            <td style="font-size:12px;white-space:nowrap;">${formatTime(t.created_at)}</td>
            <td style="font-size:12px;color:#999;">${duration}</td>
            <td style="white-space:nowrap;">
                <button class="btn btn-secondary" onclick="showUnifiedTaskDetail('${t.id}','${t.type}')" style="padding:3px 8px;font-size:11px;">详情</button>
                ${t.status === 'running' && t.type === 'offline_audio' ? `<button class="btn btn-danger" onclick="cancelTask('${t.id}')" style="padding:3px 8px;font-size:11px;margin-left:3px;">取消</button>` : ''}
                ${['completed', 'failed', 'cancelled'].includes(t.status) && t.type === 'offline_audio' ? `<button class="btn btn-danger" onclick="deleteSingleTask('${t.id}', 'offline')" style="padding:3px 8px;font-size:11px;margin-left:3px;background:#999;color:white;">删除</button>` : ''}
                ${['completed', 'failed'].includes(t.status) && t.type === 'online' ? `<button class="btn btn-danger" onclick="deleteSingleTask('${t.id.replace('online_','')}', 'online')" style="padding:3px 8px;font-size:11px;margin-left:3px;background:#999;color:white;">删除</button>` : ''}
            </td>
        </tr>
    `;
}

function getTypeBadge(type) {
    const map = {
        'offline_audio': '<span class="badge badge-secondary" style="background:#e8e8e8;color:#666;">离线</span>',
        'online': '<span class="badge" style="background:#e6f7ff;color:#1890ff;border:1px solid #91d5ff;">在线</span>',
        'custom_detection': '<span class="badge" style="background:#fff7e6;color:#fa8c16;border:1px solid #ffd591;">图片检测</span>',
        'training': '<span class="badge" style="background:#f9f0ff;color:#722ed1;border:1px solid #d3adf7;">训练</span>',
    };
    return map[type] || `<span class="badge badge-secondary">${type || '-'}</span>`;
}

function _familyLabel(family) {
    const map = {'dinomaly':'Dinomaly','dinomaly2':'Dinomaly2','anomalib':'Anomalib','ader':'ADer'};
    return map[family] || family;
}

// ==================== 统一任务详情 ====================

async function showUnifiedTaskDetail(taskId, taskType) {
    if (taskType === 'offline_audio') {
        await showTaskDetailModal(taskId);
    } else if (taskType === 'online') {
        const resultId = taskId.replace('online_', '');
        await showOnlineDetailModal(resultId);
    } else if (taskType === 'training') {
        await showTrainingDetailModal(taskId);
    } else if (taskType === 'custom_detection') {
        await showCustomDetectionDetailModal(taskId);
    }
}

async function showTrainingDetailModal(taskId) {
    try {
        const resp = await fetch(`${API_BASE}/api/training/status/${taskId}`);
        if (!resp.ok) throw new Error('获取训练任务详情失败');
        const data = await resp.json();

        const statusBadge = getStatusBadge(data.status);
        const statusColor = data.status === 'completed' ? '#52c41a' : data.status === 'failed' ? '#ff4d4f' : '#1890ff';

        let html = `
            <div class="task-detail-section">
                <h4>📋 训练任务详情</h4>
                <div class="task-detail-grid">
                    <div class="task-detail-item"><span class="label">任务ID</span><span class="value"><code>${taskId.substring(0, 12)}...</code></span></div>
                    <div class="task-detail-item"><span class="label">状态</span><span class="value" style="color:${statusColor}">${statusBadge}</span></div>
                    <div class="task-detail-item"><span class="label">算法族</span><span class="value">${_familyLabel(data.algorithm_family || '')}</span></div>
                    <div class="task-detail-item"><span class="label">算法</span><span class="value">${data.algorithm_name || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">训练类别</span><span class="value">${(data.categories || []).join(', ') || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">数据来源</span><span class="value">${data.data_source || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">进度</span><span class="value">${data.progress || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">模型路径</span><span class="value" style="word-break:break-all;">${data.model_path || '-'}</span></div>
                </div>
            </div>
        `;

        if (data.error) {
            html += `
                <div class="task-detail-section">
                    <h4>❌ 错误信息</h4>
                    <div style="font-size:13px;padding:8px 12px;background:#fff2f0;border:1px solid #ffccc7;border-radius:4px;color:#cf1322;">${data.error}</div>
                </div>
            `;
        }

        const modal = document.getElementById('customModal');
        document.getElementById('modalTitle').textContent = '🎯 训练任务详情';
        document.getElementById('modalMessage').innerHTML = html;
        document.getElementById('modalMessage').style.whiteSpace = 'normal';
        document.getElementById('modalFooter').innerHTML = `<button class="modal-btn modal-btn-primary" onclick="closeModal()">关闭</button>`;
        modal.classList.add('active');
    } catch (e) {
        await showModal('查看训练任务详情失败: ' + e.message, '错误');
    }
}

async function showCustomDetectionDetailModal(taskId) {
    try {
        const resp = await fetch(`${API_BASE}/api/custom-detection/result/${taskId}`);
        if (!resp.ok) throw new Error('获取检测任务详情失败');
        const data = await resp.json();

        const statusBadge = getStatusBadge(data.status);
        const statusColor = data.status === 'completed' ? '#52c41a' : data.status === 'failed' ? '#ff4d4f' : '#1890ff';
        const summary = data.summary || {};

        let html = `
            <div class="task-detail-section">
                <h4>📋 图片检测详情</h4>
                <div class="task-detail-grid">
                    <div class="task-detail-item"><span class="label">任务ID</span><span class="value"><code>${taskId.substring(0, 12)}...</code></span></div>
                    <div class="task-detail-item"><span class="label">状态</span><span class="value" style="color:${statusColor}">${statusBadge}</span></div>
                    <div class="task-detail-item"><span class="label">算法</span><span class="value">${data.algorithm || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">阈值</span><span class="value">${data.threshold || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">总图片</span><span class="value">${summary.total || '-'}</span></div>
                    <div class="task-detail-item"><span class="label">异常数</span><span class="value" style="color:${(summary.anomaly_count || 0) > 0 ? '#dc3545' : '#52c41a'}">${summary.anomaly_count || 0}</span></div>
                    <div class="task-detail-item"><span class="label">正常数</span><span class="value">${summary.normal_count || 0}</span></div>
                    <div class="task-detail-item"><span class="label">平均分数</span><span class="value">${summary.avg_anomaly_score ? summary.avg_anomaly_score.toFixed(4) : '-'}</span></div>
                </div>
            </div>
        `;

        if (data.error) {
            html += `
                <div class="task-detail-section">
                    <h4>❌ 错误信息</h4>
                    <div style="font-size:13px;padding:8px 12px;background:#fff2f0;border:1px solid #ffccc7;border-radius:4px;color:#cf1322;">${data.error}</div>
                </div>
            `;
        }

        const modal = document.getElementById('customModal');
        document.getElementById('modalTitle').textContent = '🖼️ 图片检测详情';
        document.getElementById('modalMessage').innerHTML = html;
        document.getElementById('modalMessage').style.whiteSpace = 'normal';
        document.getElementById('modalFooter').innerHTML = `<button class="modal-btn modal-btn-primary" onclick="closeModal()">关闭</button>`;
        modal.classList.add('active');
    } catch (e) {
        await showModal('查看检测任务详情失败: ' + e.message, '错误');
    }
}

// ==================== 筛选与分页 ====================

function filterByStatus(status) {
    currentStatusFilter = status;
    currentPage = 1;

    document.querySelectorAll('#taskStats .stat-card').forEach(card => {
        card.classList.toggle('active-filter', card.dataset.filter === status);
    });

    document.getElementById('filterStatus').value = status;
    loadTaskList();
}

function applyFilters() {
    currentPage = 1;
    currentStatusFilter = '';
    currentTypeFilter = document.getElementById('filterType')?.value || 'all';

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

function populateAlgorithmFilter(tasks) {
    const select = document.getElementById('filterAlgorithm');
    if (select.options.length > 1) return;

    const algos = new Set();
    tasks.forEach(t => {
        if (t.algorithm) algos.add(t.algorithm);
        if (t.algorithm_family) algos.add(t.algorithm_family);
    });
    algos.forEach(algo => {
        const opt = document.createElement('option');
        opt.value = algo;
        opt.textContent = _familyLabel(algo) || algo;
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
    if (task.type === 'online') return '—';
    if (task.status === 'pending' || task.status === 'running') {
        if (task.started_at) return `进行中 ${getTimeElapsed(task.started_at)}`;
        return '等待中';
    }
    if (task.completed_at && task.started_at) {
        return getTimeDiff(task.started_at, task.completed_at);
    }
    return '-';
}

function getTimeElapsed(timeStr) {
    if (!timeStr) return '—';
    // 支持时间戳和 ISO 格式
    const start = typeof timeStr === 'number' ? timeStr * 1000 : new Date(timeStr).getTime();
    const diff = Math.floor((Date.now() - start) / 1000);
    return formatDuration(diff);
}

function getTimeDiff(startStr, endStr) {
    if (!startStr || !endStr) return '-';
    const start = typeof startStr === 'number' ? startStr * 1000 : new Date(startStr).getTime();
    const end = typeof endStr === 'number' ? endStr * 1000 : new Date(endStr).getTime();
    const diff = Math.floor((end - start) / 1000);
    return formatDuration(diff);
}

function formatDuration(seconds) {
    if (seconds < 0) return '-';
    if (seconds < 60) return `${seconds}秒`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分${seconds % 60}秒`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}时${m}分`;
}

function formatTime(timeStr) {
    if (!timeStr) return '-';
    try {
        // 支持时间戳和 ISO 格式
        const date = typeof timeStr === 'number' ? new Date(timeStr * 1000) : new Date(timeStr);
        return date.toLocaleString('zh-CN', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return timeStr;
    }
}

// ==================== 离线任务详情 ====================

async function showTaskDetailModal(taskId) {
    try {
        const response = await fetch(`${API_BASE}/api/detection/result/${taskId}`);
        if (!response.ok) throw new Error('获取任务详情失败');
        const data = await response.json();

        renderTaskDetailHtml(taskId, data);
    } catch (error) {
        console.error('查看任务失败:', error);
        await showModal('查看任务详情失败: ' + error.message, '错误');
    }
}

function renderTaskDetailHtml(taskId, data) {
    const statusBadge = getStatusBadge(data.status);
    const statusColor = data.status === 'completed' ? '#52c41a' : data.status === 'failed' ? '#ff4d4f' : '#1890ff';
    const progressColor = data.progress >= 100 ? '#52c41a' : '#667eea';

    let html = `
        <div class="task-detail-section">
            <h4>📋 基本信息</h4>
            <div class="task-detail-grid">
                <div class="task-detail-item"><span class="label">类型</span><span class="value"><span class="badge badge-secondary" style="background:#e8e8e8;color:#666;">离线检测</span></span></div>
                <div class="task-detail-item"><span class="label">任务ID</span><span class="value"><code>${taskId.substring(0, 12)}...</code></span></div>
                <div class="task-detail-item"><span class="label">状态</span><span class="value" style="color:${statusColor}">${statusBadge}</span></div>
                <div class="task-detail-item"><span class="label">算法</span><span class="value">${data.algorithm || '-'}</span></div>
                <div class="task-detail-item"><span class="label">设备</span><span class="value">${data.device || '-'}</span></div>
                <div class="task-detail-item"><span class="label">进度</span><span class="value">${data.progress?.toFixed(1) || 0}%</span></div>
                <div class="task-detail-item"><span class="label">文件数</span><span class="value">${data.results?.length || 0} 个</span></div>
            </div>
        </div>
    `;

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
        html += `
            <div class="task-detail-section">
                <h4>📊 检测结果 <span style="font-size:12px;font-weight:400;color:#999;">${data.results.length} 条，${anomalyCount} 个异常</span></h4>
                <div class="task-detail-results">
        `;
        data.results.forEach(r => {
            const isAnomaly = r.is_anomaly;
            const isFailed = r.status === '预处理失败';
            const resultIcon = isFailed ? '⚠️' : (isAnomaly ? '🔴' : '🟢');
            const scoreText = r.anomaly_score !== undefined ? r.anomaly_score.toFixed(4) : '-';
            html += `
                <div class="task-detail-result-item">
                    <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px;" title="${r.filename}">${r.filename}</span>
                    ${r.segment_index !== undefined ? `<span style="color:#999;font-size:11px;margin-left:6px;">#${r.segment_index}</span>` : ''}
                    ${r.music_name ? `<span style="color:#666;font-size:11px;margin-left:6px;">🎵 ${r.music_name}</span>` : ''}
                    <span style="color:${isFailed ? '#ffc107' : (isAnomaly ? '#dc3545' : '#52c41a')};font-weight:500;white-space:nowrap;margin-left:auto;">
                        ${resultIcon} ${isFailed ? '失败' : (isAnomaly ? '异常' : '正常')} <span style="color:#999;font-weight:400;">分数: ${scoreText}</span>
                    </span>
                </div>
            `;
        });
        html += `</div></div>`;
    }

    const modal = document.getElementById('customModal');
    document.getElementById('modalTitle').textContent = '📋 离线检测详情';
    document.getElementById('modalMessage').innerHTML = html;
    document.getElementById('modalMessage').style.whiteSpace = 'normal';
    document.getElementById('modalFooter').innerHTML = `<button class="modal-btn modal-btn-primary" onclick="closeModal()">关闭</button>`;
    modal.classList.add('active');
}

// ==================== 在线检测详情 ====================

async function showOnlineDetailModal(resultId) {
    try {
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

    let html = `
        <div class="task-detail-section">
            <h4>📋 在线检测详情</h4>
            <div class="task-detail-grid">
                <div class="task-detail-item"><span class="label">客户端</span><span class="value">${result.client_name || '-'}</span></div>
                <div class="task-detail-item"><span class="label">算法</span><span class="value">${result.algorithm || '-'}</span></div>
                <div class="task-detail-item"><span class="label">文件名</span><span class="value" style="word-break:break-all;">${result.filename || '-'}</span></div>
                <div class="task-detail-item"><span class="label">时间</span><span class="value">${formatTime(result.timestamp)}</span></div>
            </div>
        </div>
        <div class="task-detail-section">
            <h4>🎯 检测结果</h4>
            <div style="padding:12px 16px;background:#f8f9fa;border-radius:6px;">
                <span style="font-size:16px;font-weight:600;color:${statusColor};">${statusIcon} ${statusText}</span>
                <span style="font-size:13px;color:#666;margin-left:12px;">异常分数: <strong>${scoreText}</strong></span>
            </div>
        </div>
    `;

    const overlayPath = result.overlay_path;
    if (overlayPath) {
        html += `
            <div class="task-detail-section">
                <h4>📊 频谱图</h4>
                <div style="text-align:center;margin-top:8px;">
                    <img src="${API_BASE}/${overlayPath}" alt="频谱图" style="max-width:100%;max-height:300px;border-radius:6px;border:1px solid #eee;" onerror="this.style.display='none'">
                </div>
            </div>
        `;
    }

    const modal = document.getElementById('customModal');
    document.getElementById('modalTitle').textContent = '🌐 在线检测详情';
    document.getElementById('modalMessage').innerHTML = html;
    document.getElementById('modalMessage').style.whiteSpace = 'normal';
    document.getElementById('modalFooter').innerHTML = `<button class="modal-btn modal-btn-primary" onclick="closeModal()">关闭</button>`;
    modal.classList.add('active');
}

// ==================== 操作函数 ====================

async function cancelTask(taskId) {
    const confirmed = await showModal('确定要取消该任务吗？', '确认', 'confirm');
    if (!confirmed) return;

    try {
        const response = await fetch(`${API_BASE}/api/detection/cancel/${taskId}`, { method: 'POST' });
        if (!response.ok) throw new Error('取消任务失败');
        await showModal('任务已取消', '提示');
        loadTaskList();
    } catch (error) {
        console.error('取消任务失败:', error);
        await showModal('取消任务失败: ' + error.message, '错误');
    }
}

async function deleteSingleTask(taskIdOrResultId, type) {
    if (type === 'online') {
        const confirmed = await showModal('确定要删除该在线检测结果吗？', '确认删除', 'danger');
        if (!confirmed) return;

        try {
            const response = await fetch(`${API_BASE}/api/client/results/${taskIdOrResultId}`, { method: 'DELETE' });
            if (!response.ok) throw new Error('删除结果失败');
            await showModal('在线检测结果已删除', '提示');
            loadTaskList();
        } catch (error) {
            console.error('删除在线结果失败:', error);
            await showModal('删除在线结果失败: ' + error.message, '错误');
        }
    } else {
        const confirmed = await showModal('确定要删除该任务记录吗？\n（不会删除关联的文件）', '确认删除', 'danger');
        if (!confirmed) return;

        try {
            const response = await fetch(`${API_BASE}/api/tasks/${taskIdOrResultId}`, { method: 'DELETE' });
            if (!response.ok) throw new Error('删除任务失败');
            await showModal('任务已删除', '提示');
            loadTaskList();
        } catch (error) {
            console.error('删除任务失败:', error);
            await showModal('删除任务失败: ' + error.message, '错误');
        }
    }
}

async function retryTask(taskId) {
    await showModal('重试功能：请切换到"💻 离线检测"页面，重新上传文件进行检测。\n\n当前暂不支持一键重试失败任务。', '提示');
}

async function cleanupOldTasks() {
    const confirmed = await showModal(
        '确定要清理所有已完成/失败/取消的离线任务记录及相关文件吗？\n\n⚠️ 此操作不可恢复！\n\n（注：在线检测结果和训练任务不受影响）',
        '确认清理',
        'danger'
    );
    if (!confirmed) return;

    try {
        const response = await fetch(`${API_BASE}/api/tasks/cleanup?clear_all=true&include_files=true`, { method: 'POST' });
        if (!response.ok) throw new Error('清理任务失败');
        const result = await response.json();
        const stats = result.file_stats || {};
        const message = `清理完成！\n\n任务记录: ${result.removed_count || 0} 个\n上传文件: ${stats.uploads || 0} 个\n临时切片: ${stats.slice || 0} 个\n热力图: ${stats.visualize || 0} 个\n导出文件: ${stats.exports || 0} 个`;
        await showModal(message, '清理完成');
        loadTaskList();
    } catch (error) {
        console.error('清理任务失败:', error);
        await showModal('清理任务失败: ' + error.message, '错误');
    }
}
