// ==================== 任务管理模块 ====================

// 分页和筛选状态
let currentPage = 1;
const PAGE_SIZE = 10;
let totalTasks = 0;
let totalPages = 1;
let currentStatusFilter = '';
let cachedTasks = [];  // 当前页数据缓存

// ==================== 统计信息 ====================

async function loadTaskStats() {
    try {
        const response = await fetch(`${API_BASE}/api/tasks/stats`);
        if (!response.ok) return;
        const data = await response.json();

        document.getElementById('statTotal').textContent = data.total || 0;
        document.getElementById('statRunning').textContent = data.running || 0;
        document.getElementById('statQueued').textContent = data.queued || 0;
        document.getElementById('statCompleted').textContent = data.completed || 0;
        document.getElementById('statFailed').textContent = data.failed || 0;
    } catch (error) {
        console.error('加载任务统计失败:', error);
    }
}

// ==================== 任务列表 ====================

async function loadTaskList() {
    try {
        // 加载统计
        await loadTaskStats();

        // 构建查询参数
        const params = new URLSearchParams();
        const statusFilter = document.getElementById('filterStatus').value || currentStatusFilter;
        if (statusFilter) params.set('status', statusFilter);
        params.set('limit', String(PAGE_SIZE));
        params.set('offset', String((currentPage - 1) * PAGE_SIZE));

        const response = await fetch(`${API_BASE}/api/tasks/list?${params}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        totalTasks = data.total || 0;
        totalPages = Math.max(1, Math.ceil(totalTasks / PAGE_SIZE));

        const tbody = document.querySelector('#taskList tbody');

        // 搜索过滤（前端）
        const searchText = document.getElementById('filterSearch').value.trim().toLowerCase();
        let tasks = data.tasks || [];
        if (searchText) {
            tasks = tasks.filter(t => t.id.toLowerCase().includes(searchText));
        }

        // 算法过滤（前端）
        const algoFilter = document.getElementById('filterAlgorithm').value;
        if (algoFilter) {
            tasks = tasks.filter(t => t.algorithm === algoFilter);
        }

        cachedTasks = tasks;

        // 更新任务计数信息
        document.getElementById('taskCountInfo').textContent =
            `共 ${totalTasks} 个任务，当前第 ${currentPage}/${totalPages} 页`;

        if (tasks.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #999; padding: 40px;">暂无任务</td></tr>';
            document.getElementById('taskPagination').style.display = 'none';
            return;
        }

        tbody.innerHTML = tasks.map(t => {
            const statusBadge = getStatusBadge(t.status);
            const duration = getTaskDuration(t);
            const progressColor = t.progress >= 100 ? '#52c41a' : '#667eea';

            return `
                <tr>
                    <td><code title="${t.id}" style="font-size:12px;cursor:pointer;" onclick="showTaskDetailModal('${t.id}')">${t.id.substring(0, 8)}...</code></td>
                    <td><span class="badge badge-${getStatusColor(t.status)}">${statusBadge}</span></td>
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
                        ${['completed', 'failed', 'cancelled'].includes(t.status) ? `<button class="btn btn-danger" onclick="deleteSingleTask('${t.id}')" style="padding:3px 8px;font-size:11px;margin-left:3px;background:#999;color:white;">删除</button>` : ''}
                    </td>
                </tr>
            `;
        }).join('');

        renderPagination();

        // 填充算法筛选下拉
        populateAlgorithmFilter(tasks);
    } catch (error) {
        console.error('加载任务列表失败:', error);
        const tbody = document.querySelector('#taskList tbody');
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #ff4d4f; padding: 40px;">加载失败: ${error.message}</td></tr>`;
    }
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

// ==================== 任务详情 ====================

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

async function deleteSingleTask(taskId) {
    const confirmed = await showModal('确定要删除该任务记录吗？\n（不会删除关联的文件）', '确认删除', 'danger');
    if (!confirmed) return;

    try {
        const response = await fetch(`${API_BASE}/api/tasks/${taskId}`, {
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

async function retryTask(taskId) {
    // 重试实际上就是重新提交相同的文件进行检测
    // 目前可以通过引导用户到离线检测页重新上传来实现
    await showModal('重试功能：请切换到"💻 离线检测"页面，重新上传文件进行检测。\n\n当前暂不支持一键重试失败任务。', '提示');
}

async function cleanupOldTasks() {
    const confirmed = await showModal(
        '确定要清理所有已完成/失败/取消的任务记录及相关文件吗？\n\n这将删除：\n• 已完成/失败/取消的任务记录\n• 上传的音频文件 (uploads/)\n• 临时切片文件 (output/slices/)\n• 生成的热力图 (output/vis/)\n• 导出的Excel/Zip文件 (output/exports/)\n\n⚠️ 此操作不可恢复！',
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
        const message = `清理完成！\n\n任务记录: ${result.removed_count || 0} 个\n上传文件: ${stats.uploads || 0} 个\n临时切片: ${stats.slice || 0} 个\n热力图: ${stats.visualize || 0} 个\n导出文件: ${stats.exports || 0} 个`;
        await showModal(message, '清理完成');
        loadTaskList();
    } catch (error) {
        console.error('清理任务失败:', error);
        await showModal('清理任务失败: ' + error.message, '错误');
    }
}
