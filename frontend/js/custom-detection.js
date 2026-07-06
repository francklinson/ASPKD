/**
 * 自定义异常检测 - 图片异常检测
 * 支持 Dinomaly、Anomalib、BaseASD 等所有适用于图片的算法
 */

// ============ 全局状态 ============
// currentTaskId 在 app.js 中声明
let pollingInterval = null;
let currentResults = [];
let selectedHeatmapIndex = -1;

// ============ 页面初始化 ============

function initCustomDetectionPage() {
    console.log('[CustomDetection] 初始化页面...');
    // 不使用缓存标志，每次切换 Tab 都重新加载
    loadCustomAlgorithms();
    loadCustomDatasets();
    setupCustomDropZone();
}

// DOM 加载完成后自动初始化（以防 switchTab 未触发）
document.addEventListener('DOMContentLoaded', function() {
    const tab = document.getElementById('custom-detection');
    if (tab && tab.classList.contains('active')) {
        initCustomDetectionPage();
    }
});

// window load 后如果 select 还是"加载中"，重试一次
window.addEventListener('load', function() {
    const select = document.getElementById('custom-algorithm');
    if (select && select.innerHTML.includes('加载中')) {
        console.log('[CustomDetection] load 事件触发，重新初始化...');
        initCustomDetectionPage();
    }
});

// ============ 算法加载 ============
async function loadCustomAlgorithms() {
    const select = document.getElementById('custom-algorithm');
    if (!select) {
        console.warn('[CustomDetection] #custom-algorithm 未找到');
        return;
    }

    try {
        console.log('[CustomDetection] 加载算法列表...');
        const resp = await fetch('/api/custom-detection/algorithms/detail');

        if (!resp.ok) {
            const errText = await resp.text().catch(() => '未知错误');
            throw new Error(`HTTP ${resp.status}: ${errText.slice(0, 100)}`);
        }

        const data = await resp.json();
        console.log(`[CustomDetection] 算法列表加载完成: ${data.flat?.length || 0} 个`);

        select.innerHTML = '';

        // 按分组添加 optgroup
        const groupOrder = ['Dinomaly', 'Anomalib', 'BaseASD', 'MuSc (零样本)', 'SubspaceAD (少样本)'];
        let hasOptions = false;

        for (const groupName of groupOrder) {
            const algs = data.groups?.[groupName];
            if (!algs || algs.length === 0) continue;

            const group = document.createElement('optgroup');
            group.label = `── ${groupName} ──`;

            algs.forEach(alg => {
                const option = document.createElement('option');
                option.value = alg.id;
                option.textContent = alg.name || alg.id;
                group.appendChild(option);
            });

            select.appendChild(group);
            hasOptions = true;
        }

        if (!hasOptions) {
            select.innerHTML = '<option>没有可用算法</option>';
        }

        console.log(`[CustomDetection] 加载了 ${data.flat?.length || 0} 个算法`);
    } catch (e) {
        console.error('[CustomDetection] 加载算法列表失败:', e);
        const select = document.getElementById('custom-algorithm');
        if (select) {
            select.innerHTML = `<option>⚠️ 加载失败: ${e.message}</option>`;
        }
    }
}

// ============ 数据集加载 ============
async function loadCustomDatasets() {
    const container = document.getElementById('custom-dataset-list');
    if (!container) return;

    try {
        console.log('[CustomDetection] 加载数据集列表...');
        const resp = await fetch('/api/custom-detection/datasets');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const data = await resp.json();
        container.innerHTML = '';

        if (!data.datasets || data.datasets.length === 0) {
            container.innerHTML = '<div class="empty-hint">暂无可用数据集</div>';
            return;
        }

        data.datasets.forEach(ds => {
            const card = document.createElement('div');
            card.className = 'dataset-card';
            card.innerHTML = `
                <div class="dataset-card-header">
                    <span class="dataset-name">📁 ${ds.name}</span>
                    <span class="dataset-count">${ds.image_count} 张图片</span>
                </div>
            `;

            // 缩略图预览（前5张）
            if (ds.images && ds.images.length > 0) {
                const thumbs = document.createElement('div');
                thumbs.className = 'dataset-thumbs';
                ds.images.slice(0, 5).forEach(img => {
                    const thumb = document.createElement('div');
                    thumb.className = 'dataset-thumb';
                    thumb.innerHTML = `<img src="${img.url || `/data/spk/${img.category}/test/${img.label === 'good' ? 'good' : 'anomaly'}/${img.filename}`}"
                                        alt="${img.filename}" title="${img.filename}\n标签: ${img.label}">`;
                    thumbs.appendChild(thumb);
                });
                card.appendChild(thumbs);
            }

            // 使用此数据集按钮
            const btnBar = document.createElement('div');
            btnBar.className = 'dataset-actions';
            const useBtn = document.createElement('button');
            useBtn.className = 'btn btn-small';
            useBtn.textContent = '选择图片进行检测';
            useBtn.onclick = () => openDatasetBrowser(ds);
            btnBar.appendChild(useBtn);
            card.appendChild(btnBar);

            container.appendChild(card);
        });
    } catch (e) {
        console.error('[CustomDetection] 加载数据集失败:', e);
    }
}

// ============ 数据集图片浏览器 ============
function openDatasetBrowser(dataset) {
    const modal = document.getElementById('custom-dataset-modal');
    const list = document.getElementById('custom-dataset-images');
    list.innerHTML = '';

    // 更新标题
    document.getElementById('custom-dataset-modal-title').textContent =
        `选择图片 - ${dataset.name} (${dataset.image_count} 张)`;

    // 渲染图片网格（带复选框）
    dataset.images.forEach(img => {
        const item = document.createElement('div');
        item.className = 'dataset-image-item';
        // 构建可直接访问的图片 URL
        const imgUrl = img.url || `/data/spk/${img.category}/test/${img.label === 'good' ? 'good' : 'anomaly'}/${encodeURIComponent(img.filename)}`;
        item.innerHTML = `
            <label>
                <input type="checkbox" value="${img.path}" data-label="${img.label}">
                <img src="${imgUrl}" alt="${img.filename}" title="${img.filename}\n标签: ${img.label}">
                <span class="img-label ${img.label}">${img.label}</span>
            </label>
        `;
        list.appendChild(item);
    });

    // 选择按钮
    document.getElementById('custom-dataset-select-btn').onclick = () => {
        const checked = list.querySelectorAll('input[type="checkbox"]:checked');
        const paths = Array.from(checked).map(cb => cb.value);
        if (paths.length === 0) {
            alert('请至少选择1张图片');
            return;
        }
        modal.classList.remove('active');
        startDatasetDetection(dataset.name, paths);
    };

    modal.classList.add('active');

    // 全选/取消
    document.getElementById('custom-dataset-select-all').onclick = () => {
        const allCbs = list.querySelectorAll('input[type="checkbox"]');
        const isChecked = allCbs.length > 0 && !allCbs[0].checked;
        allCbs.forEach(cb => cb.checked = isChecked);
    };
}

// ============ 从数据集启动检测 ============
async function startDatasetDetection(datasetName, imagePaths) {
    const algorithm = document.getElementById('custom-algorithm').value;
    const threshold = parseFloat(document.getElementById('custom-threshold').value) || 0.5;

    showCustomStatus(`正在准备检测... 算法: ${algorithm}, 图片: ${imagePaths.length} 张`, 'info');

    try {
        const resp = await fetch('/api/custom-detection/from-dataset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                dataset: datasetName,
                image_paths: imagePaths,
                algorithm: algorithm,
                threshold: threshold
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '创建任务失败');
        }

        const data = await resp.json();
        currentTaskId = data.task_id;
        startPolling(data.task_id);
        showCustomStatus(`✅ 任务已创建 (${data.task_id.slice(0, 8)}...)，等待处理...`, 'info');

    } catch (e) {
        console.error('[CustomDetection] 启动检测失败:', e);
        showCustomStatus(`❌ 启动检测失败: ${e.message}`, 'error');
    }
}

// ============ 拖放上传区 ============
function setupCustomDropZone() {
    const dropZone = document.getElementById('custom-dropzone');
    const fileInput = document.getElementById('custom-file-input');

    if (!dropZone || !fileInput) return;

    // 点击上传
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleCustomFiles(e.target.files);
        }
    });

    // 拖放
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleCustomFiles(e.dataTransfer.files);
        }
    });
}

function handleCustomFiles(files) {
    const validFiles = Array.from(files).filter(f =>
        /\.(png|jpg|jpeg|bmp|webp)$/i.test(f.name)
    );

    if (validFiles.length === 0) {
        alert('请上传图片文件 (png/jpg/jpeg/bmp/webp)');
        return;
    }

    // 显示选中文件列表
    const fileList = document.getElementById('custom-file-list');
    fileList.innerHTML = '';
    validFiles.forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <span class="file-icon">🖼️</span>
            <span class="file-name">${f.name}</span>
            <span class="file-size">${(f.size / 1024).toFixed(1)} KB</span>
        `;
        fileList.appendChild(item);
    });

    // 更新上传按钮状态
    const uploadBtn = document.getElementById('custom-upload-btn');
    uploadBtn.disabled = false;
    uploadBtn.dataset.files = validFiles.length;

    // 存储文件引用
    uploadBtn._files = validFiles;

    showCustomStatus(`已选择 ${validFiles.length} 张图片，点击"开始检测"`, 'info');
}

// ============ 上传并检测 ============
async function startCustomDetection() {
    const uploadBtn = document.getElementById('custom-upload-btn');
    const files = uploadBtn._files;

    if (!files || files.length === 0) {
        alert('请先选择图片文件');
        return;
    }

    const algorithm = document.getElementById('custom-algorithm').value;
    const threshold = parseFloat(document.getElementById('custom-threshold').value) || 0.5;

    uploadBtn.disabled = true;
    uploadBtn.textContent = '⏳ 上传中...';
    showCustomStatus(`正在上传 ${files.length} 张图片...`, 'info');

    try {
        const formData = new FormData();
        files.forEach(f => formData.append('files', f));
        formData.append('algorithm', algorithm);
        formData.append('threshold', threshold);

        const resp = await fetch('/api/custom-detection/upload', {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '上传失败');
        }

        const data = await resp.json();
        currentTaskId = data.task_id;

        // 清空文件列表
        document.getElementById('custom-file-list').innerHTML = '';
        uploadBtn._files = null;
        uploadBtn.disabled = false;
        uploadBtn.textContent = '🚀 开始检测';

        showCustomStatus(`✅ 任务已创建，等待处理... (${data.task_id.slice(0, 8)}...)`, 'info');
        startPolling(data.task_id);

    } catch (e) {
        console.error('[CustomDetection] 上传失败:', e);
        showCustomStatus(`❌ 上传失败: ${e.message}`, 'error');
        uploadBtn.disabled = false;
        uploadBtn.textContent = '🚀 开始检测';
    }
}

// ============ 轮询结果 ============
function startPolling(taskId) {
    // 显示进度条
    document.getElementById('custom-progress-container').style.display = 'block';
    document.getElementById('custom-results').style.display = 'none';

    if (pollingInterval) {
        clearInterval(pollingInterval);
    }

    pollingInterval = setInterval(async () => {
        try {
            const resp = await fetch(`/api/custom-detection/result/${taskId}`);
            if (!resp.ok) {
                if (resp.status === 404) {
                    stopPolling();
                    showCustomStatus('❌ 任务不存在', 'error');
                    return;
                }
                return;
            }

            const data = await resp.json();

            if (data.status === 'processing' || data.status === 'queued') {
                // 更新进度
                const progress = data.progress || 0;
                document.getElementById('custom-progress-bar').style.width = `${progress}%`;
                document.getElementById('custom-progress-text').textContent =
                    data.message || `处理中... ${Math.round(progress)}%`;
                showCustomStatus(data.message || `处理中... ${Math.round(progress)}%`, 'info');

            } else if (data.status === 'completed') {
                stopPolling();
                document.getElementById('custom-progress-bar').style.width = '100%';
                document.getElementById('custom-progress-text').textContent = '✅ 检测完成';
                showCustomResults(data);

            } else if (data.status === 'failed') {
                stopPolling();
                document.getElementById('custom-progress-container').style.display = 'none';
                showCustomStatus(`❌ 检测失败: ${data.error || '未知错误'}`, 'error');
            }

        } catch (e) {
            console.error('[CustomDetection] 轮询错误:', e);
        }
    }, 1000);

    // 超时处理（10分钟后停止轮询）
    setTimeout(() => {
        if (pollingInterval) {
            stopPolling();
            showCustomStatus('⏰ 轮询超时，请手动刷新', 'warn');
        }
    }, 600000);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// ============ 展示结果 ============
function showCustomResults(data) {
    const results = data.results || [];
    const summary = data.summary || {};

    currentResults = results;

    // 隐藏进度条，显示结果区
    document.getElementById('custom-progress-container').style.display = 'none';
    const resultsContainer = document.getElementById('custom-results');
    resultsContainer.style.display = 'block';

    // 更新摘要
    document.getElementById('custom-result-algorithm').textContent = data.algorithm || '-';
    document.getElementById('custom-result-total').textContent = summary.total || 0;
    document.getElementById('custom-result-anomaly').textContent = summary.anomaly_count || 0;
    document.getElementById('custom-result-normal').textContent = summary.normal_count || 0;
    document.getElementById('custom-result-avg-score').textContent =
        summary.avg_anomaly_score ? summary.avg_anomaly_score.toFixed(4) : '-';
    document.getElementById('custom-result-avg-time').textContent =
        summary.avg_inference_time_ms ? `${summary.avg_inference_time_ms.toFixed(1)} ms` : '-';

    // 渲染结果卡片
    const grid = document.getElementById('custom-result-grid');
    grid.innerHTML = '';

    if (results.length === 0) {
        grid.innerHTML = '<div class="empty-hint">无检测结果</div>';
        return;
    }

    results.forEach((r, idx) => {
        const card = document.createElement('div');
        card.className = `result-card ${r.is_anomaly ? 'anomaly' : 'normal'}`;

        // 缩略图
        const thumbUrl = r.overlay_url || r.original_url || '';
        const heatmapAvailable = r.has_heatmap && r.heatmap_url;

        card.innerHTML = `
            <div class="result-card-img">
                ${thumbUrl ? `<img src="${thumbUrl}" alt="${r.filename}" onerror="this.style.display='none'">` :
                            '<div class="no-img">无预览</div>'}
            </div>
            <div class="result-card-info">
                <div class="result-filename" title="${r.filename}">${r.filename}</div>
                <div class="result-score">
                    分数: <span class="score-value ${r.is_anomaly ? 'high' : 'low'}">${r.anomaly_score.toFixed(4)}</span>
                </div>
                <div class="result-status ${r.is_anomaly ? 'tag-anomaly' : 'tag-normal'}">
                    ${r.is_anomaly ? '🔴 异常' : '🟢 正常'}
                </div>
                ${r.inference_time_ms ? `<div class="result-time">${r.inference_time_ms.toFixed(1)} ms</div>` : ''}
                <div class="result-actions">
                    <button class="btn btn-small" onclick="showCustomDetail(${idx})">📊 详情</button>
                </div>
            </div>
            ${r.error ? `<div class="result-error" title="${r.error}">⚠️ ${r.error}</div>` : ''}
        `;

        grid.appendChild(card);
    });

    // 更新状态栏
    const anomalyPct = summary.total > 0 ? ((summary.anomaly_count / summary.total) * 100).toFixed(1) : 0;
    showCustomStatus(
        `✅ 检测完成: ${summary.total} 张, ` +
        `异常 ${summary.anomaly_count} (${anomalyPct}%), ` +
        `正常 ${summary.normal_count}, ` +
        `算法: ${data.algorithm}`,
        'success'
    );
}

// ============ 详情模态框 ============
function showCustomDetail(index) {
    const result = currentResults[index];
    if (!result) return;

    const modal = document.getElementById('custom-detail-modal');
    modal.classList.add('active');

    // 详情内容
    const container = document.getElementById('custom-detail-content');

    // 图片切换
    let imageHtml = '<div class="detail-image-tabs">';
    if (result.original_url) {
        imageHtml += `<button class="tab-btn active" onclick="switchDetailImage(this, 'original', ${index})">原图</button>`;
    }
    if (result.overlay_url) {
        imageHtml += `<button class="tab-btn" onclick="switchDetailImage(this, 'overlay', ${index})">叠加</button>`;
    }
    if (result.heatmap_url) {
        imageHtml += `<button class="tab-btn" onclick="switchDetailImage(this, 'heatmap', ${index})">热力图</button>`;
    }
    imageHtml += '</div>';

    // 默认显示叠加图或原图
    const defaultImg = result.overlay_url || result.original_url || '';
    imageHtml += `<div class="detail-image-container">
        <img id="detail-main-img" src="${defaultImg}" alt="${result.filename}">
    </div>`;

    // 信息
    imageHtml += `
        <div class="detail-info">
            <table>
                <tr><td>文件名</td><td>${result.filename}</td></tr>
                <tr><td>异常分数</td><td class="${result.is_anomaly ? 'score-high' : 'score-low'}">${result.anomaly_score.toFixed(4)}</td></tr>
                <tr><td>判定</td><td>${result.is_anomaly ? '🔴 异常' : '🟢 正常'}</td></tr>
                <tr><td>推理时间</td><td>${result.inference_time_ms ? result.inference_time_ms.toFixed(1) + ' ms' : '-'}</td></tr>
                ${result.error ? `<tr><td>错误</td><td class="error-text">${result.error}</td></tr>` : ''}
            </table>
        </div>
    `;

    container.innerHTML = imageHtml;
}

function switchDetailImage(btn, type, index) {
    const result = currentResults[index];
    if (!result) return;

    // 切换 tab 激活状态
    btn.parentElement.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // 切换图片
    const img = document.getElementById('detail-main-img');
    if (type === 'original') img.src = result.original_url || '';
    else if (type === 'overlay') img.src = result.overlay_url || '';
    else if (type === 'heatmap') img.src = result.heatmap_url || '';
}

// 关闭详情模态框
function closeCustomDetail() {
    document.getElementById('custom-detail-modal').classList.remove('active');
}

// ============ 历史记录恢复 ============
function restoreCustomHistory() {
    const historyJson = localStorage.getItem('custom_detection_history');
    if (!historyJson) return;

    try {
        const history = JSON.parse(historyJson);
        // 显示最近3条记录
        const recent = history.slice(0, 3);
        // 目前暂不自动恢复，仅留作后续扩展
    } catch (e) {
        // ignore
    }
}

function saveCustomHistory(taskId, algorithm) {
    try {
        const historyJson = localStorage.getItem('custom_detection_history');
        const history = historyJson ? JSON.parse(historyJson) : [];
        history.unshift({ taskId, algorithm, timestamp: Date.now() });
        // 保留最近20条
        if (history.length > 20) history.length = 20;
        localStorage.setItem('custom_detection_history', JSON.stringify(history));
    } catch (e) {
        // ignore
    }
}

// ============ 辅助函数 ============
function showCustomStatus(message, type) {
    const statusEl = document.getElementById('custom-status');
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.className = `custom-status ${type || 'info'}`;
    statusEl.style.display = 'block';
}

function clearCustomResults() {
    document.getElementById('custom-results').style.display = 'none';
    document.getElementById('custom-progress-container').style.display = 'none';
    document.getElementById('custom-file-list').innerHTML = '';
    document.getElementById('custom-status').style.display = 'none';
    currentResults = [];
    currentTaskId = null;
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}
