// ==================== 参考音频库管理模块 ====================

function handleRefFile(file) {
    // 验证文件格式
    const allowedExtensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(ext)) {
        showModal('不支持的文件格式，仅支持 WAV, MP3, FLAC, AAC, OGG, M4A', '错误');
        return;
    }

    selectedRefFile = file;

    // 显示选中的文件
    document.getElementById('refSelectedFile').style.display = 'block';
    document.getElementById('refFileName').textContent = '📄 ' + file.name;
    document.getElementById('refFileSize').textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';

    // 启用添加按钮
    document.getElementById('addRefBtn').disabled = false;

    // 如果没有填写名称，自动填充文件名
    const nameInput = document.getElementById('refAudioName');
    if (!nameInput.value) {
        nameInput.value = file.name.replace(/\.[^/.]+$/, '');
    }
}

// 添加参考音频
async function addReferenceAudio() {
    if (!selectedRefFile) {
        await showModal('请先选择音频文件', '提示');
        return;
    }

    const btn = document.getElementById('addRefBtn');
    const progress = document.getElementById('refProgress');
    const progressFill = document.getElementById('refProgressFill');
    const progressText = document.getElementById('refProgressText');

    btn.disabled = true;
    btn.textContent = '⏳ 上传中...';
    progress.style.display = 'block';
    progressFill.style.width = '50%';
    progressText.textContent = '正在上传文件...';

    try {
        const formData = new FormData();
        formData.append('file', selectedRefFile);

        const name = document.getElementById('refAudioName').value.trim();
        if (name) {
            formData.append('name', name);
        }

        progressFill.style.width = '70%';
        progressText.textContent = '正在生成指纹...';

        const response = await fetch(`${API_BASE}/api/reference-audio/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            progressFill.style.width = '100%';
            progressText.textContent = '完成!';

            // 清除缓存，强制刷新
            referenceListCache = null;
            referenceStatsCache = null;

            await showModal(result.message, '添加成功');

            // 重置表单
            selectedRefFile = null;
            document.getElementById('refFileInput').value = '';
            document.getElementById('refAudioName').value = '';
            document.getElementById('refSelectedFile').style.display = 'none';
            document.getElementById('addRefBtn').disabled = true;
            document.getElementById('addRefBtn').textContent = '➕ 添加到指纹库';
            progress.style.display = 'none';

            // 强制刷新列表
            loadReferenceList(true);
            loadReferenceStats(true);

            // 刷新离线检测页面的参考音频列表
            loadReferenceAudios();
            console.log('[参考音频] 已刷新离线检测页面的参考音频列表');
        } else {
            throw new Error(result.detail || '添加失败');
        }
    } catch (error) {
        console.error('添加参考音频失败:', error);
        await showModal('添加失败: ' + error.message, '错误');

        btn.disabled = false;
        btn.textContent = '➕ 添加到指纹库';
        progress.style.display = 'none';
    }
}

// 参考音频列表缓存
let referenceListCache = null;
let referenceListCacheTime = 0;
const REFERENCE_CACHE_TTL = 30000; // 缓存30秒
let isLoadingReferenceList = false;

// 加载参考音频列表
async function loadReferenceList(forceRefresh = false) {
    // 防止重复加载
    if (isLoadingReferenceList) {
        console.log('[Reference] 列表正在加载中，跳过重复请求');
        return;
    }

    // 检查缓存
    const now = Date.now();
    if (!forceRefresh && referenceListCache && (now - referenceListCacheTime) < REFERENCE_CACHE_TTL) {
        console.log('[Reference] 使用缓存数据');
        renderReferenceList(referenceListCache);
        return;
    }

    isLoadingReferenceList = true;

    // 显示加载状态
    const tbody = document.querySelector('#referenceList tbody');
    const emptyState = document.getElementById('refEmptyState');
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 40px;">
        <div style="color: #667eea;">⏳ 加载中...</div>
    </td></tr>`;
    emptyState.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/api/reference-audio/list`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // 更新缓存
        referenceListCache = data;
        referenceListCacheTime = now;

        renderReferenceList(data);
    } catch (error) {
        console.error('加载参考音频列表失败:', error);
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #ff4d4f; padding: 40px;">加载失败: ${error.message}</td></tr>`;
    } finally {
        isLoadingReferenceList = false;
    }
}

// 渲染参考音频列表
function renderReferenceList(data) {
    const tbody = document.querySelector('#referenceList tbody');
    const emptyState = document.getElementById('refEmptyState');

    if (!data.references || data.references.length === 0) {
        tbody.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }

    emptyState.style.display = 'none';
    tbody.innerHTML = data.references.map(ref => `
        <tr>
            <td>${ref.music_id}</td>
            <td>${escapeHtml(ref.name)}</td>
            <td>${ref.hash_count.toLocaleString()}</td>
            <td>
                <button class="play-btn" onclick="toggleRefAudioPlay(${ref.music_id}, this)"
                        title="点击试听"
                        style="background:#6366f1;color:white;border:none;border-radius:4px;padding:4px 10px;font-size:12px;cursor:pointer;min-width:58px;text-align:center;">
                    ▶ 试听
                </button>
            </td>
            <td>
                <button class="btn btn-danger" onclick="deleteReferenceAudio(${ref.music_id}, '${escapeHtml(ref.name).replace(/'/g, "\\'")}')" style="padding: 4px 8px; font-size: 12px;">
                    🗑️ 删除
                </button>
            </td>
        </tr>
    `).join('');
}

// HTML转义函数
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 参考音频统计缓存
let referenceStatsCache = null;
let referenceStatsCacheTime = 0;
const REFERENCE_STATS_CACHE_TTL = 30000; // 缓存30秒

// 加载参考音频统计
async function loadReferenceStats(forceRefresh = false) {
    // 检查缓存
    const now = Date.now();
    if (!forceRefresh && referenceStatsCache && (now - referenceStatsCacheTime) < REFERENCE_STATS_CACHE_TTL) {
        console.log('[Reference] 使用缓存的统计信息');
        renderReferenceStats(referenceStatsCache);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/reference-audio/stats`);
        if (!response.ok) return;

        const data = await response.json();

        // 更新缓存
        referenceStatsCache = data;
        referenceStatsCacheTime = now;

        renderReferenceStats(data);
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

// 渲染参考音频统计
function renderReferenceStats(data) {
    document.getElementById('refTotalCount').textContent = data.total_references;
    document.getElementById('refTotalHashes').textContent = data.total_hashes.toLocaleString();
    document.getElementById('refAvgHashes').textContent = Math.round(data.average_hashes).toLocaleString();
}

// 删除参考音频
async function deleteReferenceAudio(musicId, name) {
    const confirmed = await showModal(
        `确定要删除参考音频 "${name}" (ID=${musicId}) 吗？\n\n这将同时删除其指纹数据。`,
        '确认删除',
        'danger'
    );

    if (!confirmed) return;

    try {
        const response = await fetch(`${API_BASE}/api/reference-audio/${musicId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('删除失败');
        }

        const result = await response.json();

        // 清除缓存，强制刷新
        referenceListCache = null;
        referenceStatsCache = null;

        await showModal(result.message, '删除成功');

        // 强制刷新列表
        loadReferenceList(true);
        loadReferenceStats(true);
    } catch (error) {
        console.error('删除参考音频失败:', error);
        await showModal('删除失败: ' + error.message, '错误');
    }
}

// ==================== 参考音频试听功能 ====================

let refAudioPlayer = null;
let refPlayingButton = null;
const refPlayIconSvg = '▶ 试听';
const refPauseIconSvg = '⏸ 暂停';

function toggleRefAudioPlay(musicId, buttonElement) {
    // 如果点击的是当前正在播放的按钮
    if (refPlayingButton === buttonElement && refAudioPlayer) {
        if (refAudioPlayer.paused) {
            refAudioPlayer.play();
            buttonElement.textContent = refPauseIconSvg;
            buttonElement.style.background = '#4f46e5';
        } else {
            refAudioPlayer.pause();
            buttonElement.textContent = refPlayIconSvg;
            buttonElement.style.background = '#6366f1';
        }
        return;
    }

    // 停止之前播放的音频
    if (refAudioPlayer) {
        refAudioPlayer.pause();
        refAudioPlayer = null;
    }
    if (refPlayingButton) {
        refPlayingButton.textContent = refPlayIconSvg;
        refPlayingButton.style.background = '#6366f1';
    }

    // 先通过 HEAD 请求检查音频文件是否存在
    const audioUrl = `${API_BASE}/api/reference-audio/audio/${musicId}`;

    // 创建新的音频对象
    refAudioPlayer = new Audio(audioUrl);
    refPlayingButton = buttonElement;

    // 更新按钮状态
    buttonElement.style.opacity = '0.7';

    refAudioPlayer.oncanplay = function() {
        // 可以播放时自动开始
        buttonElement.textContent = refPauseIconSvg;
        buttonElement.style.background = '#4f46e5';
        refAudioPlayer.play().catch(err => {
            console.error('参考音频播放失败:', err);
            buttonElement.textContent = refPlayIconSvg;
            buttonElement.style.background = '#6366f1';
            showModal('音频播放失败，浏览器可能不支持该音频格式', '播放失败');
        });
    };

    // 音频加载出错时的处理
    refAudioPlayer.onerror = function() {
        console.error('参考音频加载出错, music_id=' + musicId);
        buttonElement.textContent = refPlayIconSvg;
        buttonElement.style.background = '#6366f1';
        refPlayingButton = null;
        refAudioPlayer = null;

        // 获取具体错误信息
        const error = refAudioPlayer?.error;
        let message = '音频文件不存在或无法访问，请确认参考音频已正确上传';
        if (error) {
            switch (error.code) {
                case MediaError.MEDIA_ERR_ABORTED:
                    message = '音频加载被中断';
                    break;
                case MediaError.MEDIA_ERR_NETWORK:
                    message = '音频加载失败（网络错误），请检查服务端音频文件是否存在';
                    break;
                case MediaError.MEDIA_ERR_DECODE:
                    message = '音频解码失败，文件可能已损坏';
                    break;
                case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
                    message = '浏览器不支持该音频格式';
                    break;
            }
        }
        showModal(message, '播放失败');
    };

    // 音频结束时的处理
    refAudioPlayer.onended = function() {
        buttonElement.textContent = refPlayIconSvg;
        buttonElement.style.background = '#6366f1';
        refPlayingButton = null;
        refAudioPlayer = null;
    };

    // 开始加载音频
    refAudioPlayer.load();
}

function initReferenceUpload() {
    const refDropzone = document.getElementById('refDropzone');
    const refFileInput = document.getElementById('refFileInput');

    if (refDropzone && refFileInput) {
        refDropzone.addEventListener('click', () => refFileInput.click());

        refDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            refDropzone.classList.add('dragover');
        });

        refDropzone.addEventListener('dragleave', () => {
            refDropzone.classList.remove('dragover');
        });

        refDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            refDropzone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleRefFile(files[0]);
            }
        });

        refFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleRefFile(e.target.files[0]);
            }
        });

        console.log('[Reference] 文件上传事件监听器已绑定');
    } else {
        console.error('[Reference] 找不到上传区域元素:', { refDropzone: !!refDropzone, refFileInput: !!refFileInput });
    }
}
