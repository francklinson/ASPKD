// ==================== 核心模块 ====================

// 登录状态检查
(function() {
    const isLoggedIn = sessionStorage.getItem('isLoggedIn');
    if (isLoggedIn !== 'true') {
        window.location.href = '/';
    }
})();

// API 基础 URL
const API_BASE = window.location.origin;

// 显示当前用户名
function displayCurrentUser() {
    const username = sessionStorage.getItem('username') || '管理员';
    document.getElementById('currentUser').textContent = username;
}

// 退出登录
async function logout() {
    const token = sessionStorage.getItem('token');
    if (token) {
        try {
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: { 'Authorization': 'Bearer ' + token }
            });
        } catch (e) {}
    }
    sessionStorage.clear();
    window.location.href = '/';
}

// 页面加载时显示用户名
displayCurrentUser();

// 状态管理
let selectedFiles = [];
let currentTaskId = null;
let wsConnection = null;
let monitorWs = null;
let isDetecting = false; // 标记是否正在检测中
let selectedRefFile = null; // 选中的参考音频文件

// 标签切换
function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(tabName).classList.add('active');

    if (tabName === 'tasks') {
        loadTaskList();
    } else if (tabName === 'reference') {
        loadReferenceList();
        loadReferenceStats();
    } else if (tabName === 'cluster') {
        initClusterPage();
    } else if (tabName === 'dataset-builder') {
        // 数据集构建面板：首次加载时刷新 iframe
        const dsIframe = document.getElementById('datasetIframe');
        if (dsIframe && !dsIframe.dataset.loaded) {
            dsIframe.dataset.loaded = 'true';
        }
    } else if (tabName === 'training') {
        // 模型训练面板：首次加载时刷新 iframe
        const trIframe = document.getElementById('trainingIframe');
        if (trIframe && !trIframe.dataset.loaded) {
            trIframe.dataset.loaded = 'true';
        }
    }
}

// ==================== 自定义弹窗 ====================

// 自定义弹窗函数
let modalResolve = null;

function showModal(message, title = '提示', type = 'alert') {
    return new Promise((resolve) => {
        modalResolve = resolve;
        const modal = document.getElementById('customModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalMessage = document.getElementById('modalMessage');
        const modalFooter = document.getElementById('modalFooter');

        modalTitle.textContent = title;
        modalMessage.textContent = message;

        if (type === 'confirm') {
            modalFooter.innerHTML = `
                <button class="modal-btn modal-btn-secondary" onclick="closeModal(false)">取消</button>
                <button class="modal-btn modal-btn-primary" onclick="closeModal(true)">确定</button>
            `;
        } else if (type === 'danger') {
            modalFooter.innerHTML = `
                <button class="modal-btn modal-btn-secondary" onclick="closeModal(false)">取消</button>
                <button class="modal-btn modal-btn-danger" onclick="closeModal(true)">确定</button>
            `;
        } else {
            modalFooter.innerHTML = `
                <button class="modal-btn modal-btn-primary" onclick="closeModal(true)">确定</button>
            `;
        }

        modal.classList.add('active');
    });
}

// 配置不匹配弹窗
let mismatchModalResolve = null;

function showMismatchModal(message, currentAlgorithm, currentDevice) {
    return new Promise((resolve) => {
        mismatchModalResolve = resolve;
        const modal = document.getElementById('customModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalMessage = document.getElementById('modalMessage');
        const modalFooter = document.getElementById('modalFooter');

        modalTitle.textContent = '⚠️ 配置冲突';
        modalMessage.textContent = message;

        modalFooter.innerHTML = `
            <button class="modal-btn modal-btn-secondary" onclick="closeMismatchModal('cancel')">取消启动</button>
            <button class="modal-btn modal-btn-primary" onclick="closeMismatchModal('sync')">🔄 同步为离线配置</button>
        `;

        modal.classList.add('active');
    });
}

function closeMismatchModal(result) {
    const modal = document.getElementById('customModal');
    modal.classList.remove('active');
    if (mismatchModalResolve) {
        mismatchModalResolve(result);
        mismatchModalResolve = null;
    }
}

function closeModal(result = true) {
    const modal = document.getElementById('customModal');
    modal.classList.remove('active');
    // 恢复 modal body 默认样式（任务详情可能会修改它）
    const modalMessage = document.getElementById('modalMessage');
    modalMessage.style.whiteSpace = '';
    modalMessage.innerHTML = '';
    if (modalResolve) {
        modalResolve(result);
        modalResolve = null;
    }
}

// ESC键关闭弹窗
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal(false);
    }
});

// ==================== 设备列表 ====================

// 动态加载可用设备列表
async function loadAvailableDevices() {
    try {
        const response = await fetch(`${API_BASE}/api/detection/devices`);
        const data = await response.json();

        if (data.devices && data.devices.length > 0) {
            // 更新离线检测设备选择框
            const deviceSelect = document.getElementById('device');
            const deviceInfo = document.getElementById('deviceInfo');
            deviceSelect.innerHTML = '';

            // 更新监控设备选择框
            const monitorDeviceSelect = document.getElementById('monitorDevice');
            const monitorDeviceInfo = document.getElementById('monitorDeviceInfo');
            monitorDeviceSelect.innerHTML = '';

            // 更新客户端监控设备选择框
            const clientDeviceSelect = document.getElementById('clientDevice');
            const clientDeviceInfo = document.getElementById('clientDeviceInfo');
            if (clientDeviceSelect) clientDeviceSelect.innerHTML = '';

            data.devices.forEach(device => {
                // 构建设备选项文本
                let optionText = device.name;
                if (device.recommended) {
                    optionText = '【推荐】' + device.name;
                }

                // 创建设备选项
                const option = document.createElement('option');
                option.value = device.id;
                option.textContent = optionText;
                option.title = device.info || '';  // 悬停提示

                // 添加到三个选择框
                deviceSelect.appendChild(option.cloneNode(true));
                monitorDeviceSelect.appendChild(option.cloneNode(true));
                if (clientDeviceSelect) clientDeviceSelect.appendChild(option);
            });

            // 添加设备信息提示
            const gpuDevices = data.devices.filter(d => d.type === 'cuda');
            const infoText = gpuDevices.length > 0
                ? `检测到 ${data.gpu_count} 张GPU计算卡，已按可用显存排序`
                : '未检测到GPU，将使用CPU运行';

            deviceInfo.textContent = infoText;
            monitorDeviceInfo.textContent = infoText;
            if (clientDeviceInfo) clientDeviceInfo.textContent = infoText;

            // 绑定设备选择事件，显示详细信息
            deviceSelect.addEventListener('change', function() {
                const selected = data.devices.find(d => d.id === this.value);
                if (selected && selected.info) {
                    deviceInfo.textContent = selected.info;
                }
            });

            monitorDeviceSelect.addEventListener('change', function() {
                const selected = data.devices.find(d => d.id === this.value);
                if (selected && selected.info) {
                    monitorDeviceInfo.textContent = selected.info;
                }
            });

            if (clientDeviceSelect) {
                clientDeviceSelect.addEventListener('change', function() {
                    const selected = data.devices.find(d => d.id === this.value);
                    if (selected && selected.info && clientDeviceInfo) {
                        clientDeviceInfo.textContent = selected.info;
                    }
                });
            }

            console.log(`[设备] 已加载 ${data.devices.length} 个设备选项`);
        }
    } catch (error) {
        console.error('加载设备列表失败:', error);
        // 使用默认选项
        const defaultOption = '<option value="auto">自动选择 (GPU优先)</option><option value="cpu">CPU</option>';
        document.getElementById('device').innerHTML = defaultOption;
        document.getElementById('monitorDevice').innerHTML = defaultOption;
        const clientDevice = document.getElementById('clientDevice');
        if (clientDevice) clientDevice.innerHTML = defaultOption;
    }
}

// ==================== 参考音频列表（离线检测） ====================

// 动态加载参考音频列表
async function loadReferenceAudios() {
    try {
        const response = await fetch(`${API_BASE}/api/detection/reference-audios`);
        const data = await response.json();

        const refSelect = document.getElementById('referenceAudio');
        const refInfo = document.getElementById('referenceAudioInfo');
        refSelect.innerHTML = '';

        if (data.references && data.references.length > 0) {
            // 添加参考音频选项
            data.references.forEach(ref => {
                const option = document.createElement('option');
                option.value = ref.path;
                option.textContent = `${ref.name} (${ref.hash_count} 个指纹点)`;
                option.title = `路径: ${ref.path}`;
                refSelect.appendChild(option);
            });

            // 检查默认参考音频是否在列表中
            const defaultInList = data.default && data.references.some(ref => ref.path === data.default);

            // 添加默认选项（仅当默认参考音频在数据库列表中时才显示）
            if (data.default && defaultInList) {
                const defaultOption = document.createElement('option');
                defaultOption.value = data.default;
                defaultOption.textContent = `默认 (${data.default.split('/').pop()})`;
                defaultOption.selected = true;
                refSelect.insertBefore(defaultOption, refSelect.firstChild);
            } else if (data.default) {
                console.log(`[参考音频] 默认参考音频 ${data.default} 不在数据库中，不显示默认选项`);
                // 默认选中第一个参考音频
                refSelect.firstChild.selected = true;
            }

            refInfo.textContent = `共 ${data.total} 个参考音频可用`;
            console.log(`[参考音频] 已加载 ${data.total} 个参考音频`);
        } else {
            // 没有参考音频时使用默认
            const defaultOption = document.createElement('option');
            defaultOption.value = data.default || '';
            defaultOption.textContent = '使用默认参考音频';
            refSelect.appendChild(defaultOption);

            if (data.error) {
                refInfo.textContent = `提示: ${data.error}`;
                refInfo.style.color = '#ff9800';
            } else {
                refInfo.textContent = '参考音频库为空，将使用默认配置';
            }
        }

        // 绑定选择事件
        refSelect.addEventListener('change', function() {
            const selectedText = this.options[this.selectedIndex].text;
            console.log(`[参考音频] 已选择: ${selectedText}`);
        });

    } catch (error) {
        console.error('加载参考音频列表失败:', error);
        const refSelect = document.getElementById('referenceAudio');
        refSelect.innerHTML = '<option value="">使用默认参考音频</option>';
        document.getElementById('referenceAudioInfo').textContent = '加载失败，将使用默认配置';
    }
}

// 页面加载时获取设备列表和参考音频列表（设备列表只需查询一次）
loadAvailableDevices();
loadReferenceAudios();
loadMonitorReferenceAudios();

// ==================== 监控页面参考音频 ====================

// 监控页面已选择的参考音频列表
let monitorSelectedReferences = [];

// 加载监控页面参考音频列表
async function loadMonitorReferenceAudios() {
    try {
        const response = await fetch(`${API_BASE}/api/detection/reference-audios`);
        const data = await response.json();

        // 更新实时监控参考音频选择框
        const refSelect = document.getElementById('monitorReferenceAudioSelect');
        if (refSelect) {
            refSelect.innerHTML = '<option value="">选择参考音频...</option>';

            if (data.references && data.references.length > 0) {
                data.references.forEach(ref => {
                    const option = document.createElement('option');
                    option.value = ref.path;
                    option.textContent = `${ref.name} (${ref.hash_count} 个指纹点)`;
                    option.title = `路径: ${ref.path}`;
                    option.dataset.name = ref.name;
                    refSelect.appendChild(option);
                });
            }
        }

        // 更新客户端监控参考音频选择框
        const clientRefSelect = document.getElementById('clientReferenceAudioSelect');
        if (clientRefSelect) {
            clientRefSelect.innerHTML = '<option value="">选择参考音频...</option>';

            if (data.references && data.references.length > 0) {
                data.references.forEach(ref => {
                    const option = document.createElement('option');
                    option.value = ref.path;
                    option.textContent = `${ref.name} (${ref.hash_count} 个指纹点)`;
                    option.title = `路径: ${ref.path}`;
                    option.dataset.name = ref.name;
                    clientRefSelect.appendChild(option);
                });
            }
        }

        console.log(`[参考音频] 已加载 ${data.total} 个参考音频（实时监控和客户端监控）`);

    } catch (error) {
        console.error('加载监控参考音频列表失败:', error);
        const refSelect = document.getElementById('monitorReferenceAudioSelect');
        if (refSelect) refSelect.innerHTML = '<option value="">加载失败</option>';
        const clientRefSelect = document.getElementById('clientReferenceAudioSelect');
        if (clientRefSelect) clientRefSelect.innerHTML = '<option value="">加载失败</option>';
    }
}

// 添加参考音频到选择列表
async function addMonitorReference() {
    const select = document.getElementById('monitorReferenceAudioSelect');
    const selectedOption = select.options[select.selectedIndex];

    if (!selectedOption || !selectedOption.value) {
        return;
    }

    const refPath = selectedOption.value;
    const refName = selectedOption.dataset.name || selectedOption.textContent.split(' (')[0];

    // 检查是否已存在
    if (monitorSelectedReferences.some(r => r.path === refPath)) {
        console.log(`[监控] 参考音频已存在: ${refName}`);
        return;
    }

    // 添加到列表
    monitorSelectedReferences.push({ path: refPath, name: refName });

    // 更新显示
    updateMonitorSelectedRefsDisplay();

    // 重置选择框
    select.value = '';

    console.log(`[监控] 添加参考音频: ${refName}`);

    // 如果监控正在运行，自动更新参考音频
    await autoUpdateMonitorReferences();
}

// 移除参考音频
async function removeMonitorReference(index) {
    const removed = monitorSelectedReferences.splice(index, 1)[0];
    updateMonitorSelectedRefsDisplay();
    console.log(`[监控] 移除参考音频: ${removed.name}`);

    // 如果监控正在运行，自动更新参考音频
    await autoUpdateMonitorReferences();
}

// 自动更新参考音频（监控运行时）
async function autoUpdateMonitorReferences() {
    // 检查监控状态
    try {
        const response = await fetch(`${API_BASE}/api/monitor/status`);
        const status = await response.json();

        if (status.is_running) {
            console.log('[监控] 检测到监控正在运行，自动更新参考音频...');
            await updateMonitorReferences();
        }
    } catch (error) {
        console.error('[监控] 检查监控状态失败:', error);
    }
}

// 更新已选参考音频显示
function updateMonitorSelectedRefsDisplay() {
    const container = document.getElementById('monitorSelectedRefs');
    const hiddenInput = document.getElementById('monitorReferenceAudios');

    if (monitorSelectedReferences.length === 0) {
        container.innerHTML = '<span style="color: #999; font-size: 12px; line-height: 20px;">未选择参考音频，将使用自动匹配</span>';
        hiddenInput.value = '';
    } else {
        container.innerHTML = monitorSelectedReferences.map((ref, index) => `
            <div style="display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; background: #e3f2fd; border: 1px solid #90caf9; border-radius: 16px; font-size: 12px; color: #1976d2;">
                <span>${ref.name}</span>
                <button type="button" onclick="removeMonitorReference(${index})" style="background: none; border: none; color: #1976d2; cursor: pointer; padding: 0; font-size: 14px; line-height: 1;">&times;</button>
            </div>
        `).join('');

        // 更新隐藏字段
        hiddenInput.value = JSON.stringify(monitorSelectedReferences.map(r => r.path));
    }
}

// 绑定添加按钮事件
document.addEventListener('DOMContentLoaded', function() {
    const addBtn = document.getElementById('addMonitorRefBtn');
    if (addBtn) {
        addBtn.addEventListener('click', addMonitorReference);
    }
});

// 更新检测上下文显示
async function updateDetectionContext() {
    try {
        const response = await fetch(`${API_BASE}/api/monitor/detection-context`);
        const context = await response.json();

        const contextDiv = document.getElementById('detectionContext');
        const contextContent = document.getElementById('contextContent');

        // 检查实时监控是否正在运行
        const isMonitorRunning = document.getElementById('monitorStatus').textContent === '运行中';

        if (context.has_running_task) {
            contextDiv.style.display = 'block';
            contextDiv.className = 'alert alert-warning';
            contextContent.innerHTML = `
                <div>🔴 有 ${context.running_count} 个离线检测任务运行中</div>
                <div style="margin-top: 4px; font-size: 13px;">
                    算法: <strong>${context.current_algorithm || '未知'}</strong> |
                    设备: <strong>${context.current_device || '未知'}</strong>
                </div>
                <div style="margin-top: 4px; font-size: 12px; color: #666;">
                    提示: 监控将强制使用与离线检测相同的算法和设备
                </div>
            `;

            // 自动同步算法和设备选择
            if (context.current_algorithm) {
                document.getElementById('monitorAlgorithm').value = context.current_algorithm;
            }
            if (context.current_device) {
                document.getElementById('monitorDevice').value = context.current_device;
            }

            // 禁用算法和设备选择
            document.getElementById('monitorAlgorithm').disabled = true;
            document.getElementById('monitorDevice').disabled = true;
        } else {
            contextDiv.style.display = 'none';
            // 只有在实时监控也未运行时才启用算法和设备选择
            if (!isMonitorRunning) {
                document.getElementById('monitorAlgorithm').disabled = false;
                document.getElementById('monitorDevice').disabled = false;
            }
        }
    } catch (error) {
        console.error('更新检测上下文失败:', error);
    }
}

// 定期刷新检测上下文
setInterval(updateDetectionContext, 3000);

// 定期刷新监控状态
setInterval(async () => {
    try {
        const response = await fetch(`${API_BASE}/api/monitor/status`);
        const status = await response.json();

        if (status.is_running) {
            document.getElementById('monitorStatus').textContent = '运行中';
            document.getElementById('monitorTotal').textContent = status.total_processed;
            document.getElementById('monitorAnomaly').textContent = status.anomaly_count;
        }
    } catch (error) {
        // 忽略错误
    }
}, 5000);

// 页面加载时检查监控状态并恢复WebSocket连接
async function initMonitorOnLoad() {
    try {
        const response = await fetch(`${API_BASE}/api/monitor/status`);
        const status = await response.json();

        if (status.is_running) {
            console.log('[Init] 检测到监控正在运行，恢复WebSocket连接');
            monitorLog('检测到监控正在运行，恢复连接...');

            // 更新UI状态
            document.getElementById('monitorStatus').textContent = '运行中';
            document.getElementById('monitorTotal').textContent = status.total_processed;
            document.getElementById('monitorAnomaly').textContent = status.anomaly_count;
            document.getElementById('startMonitorBtn').disabled = true;
            document.getElementById('stopMonitorBtn').disabled = false;

            // 禁用算法和设备选择
            document.getElementById('monitorAlgorithm').disabled = true;
            document.getElementById('monitorDevice').disabled = true;

            // 恢复监控目录显示（如果后端返回）
            if (status.directory) {
                document.getElementById('monitorDir').value = status.directory;
            }

            // 重新连接WebSocket
            connectMonitorWebSocket();
        }
    } catch (error) {
        console.error('[Init] 检查监控状态失败:', error);
    }
}

// 初始化
console.log('系统就绪，等待文件上传...');
updateDetectionContext(); // 立即执行一次
initMonitorOnLoad(); // 检查监控状态并恢复连接

// ==================== 热力图模态框 ====================

// 显示热力图大图模态框
function showHeatmapModal(heatmapUrl, filename, score) {
    const modal = document.createElement('div');
    modal.id = 'heatmapModal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        cursor: pointer;
    `;

    modal.innerHTML = `
        <div style="position: relative; max-width: 90%; max-height: 90%; background: white; border-radius: 8px; padding: 20px;" onclick="event.stopPropagation()">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <div>
                    <h3 style="margin: 0; font-size: 16px;">🔥 异常热力图</h3>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: #666;">${filename} | 异常分数: ${score.toFixed(4)}</p>
                </div>
                <button onclick="closeHeatmapModal()" style="background: none; border: none; font-size: 24px; cursor: pointer; color: #666;">&times;</button>
            </div>
            <img src="${heatmapUrl}" style="max-width: 100%; max-height: 70vh; border-radius: 4px; border: 1px solid #e0e0e0;">
            <div style="margin-top: 12px; font-size: 12px; color: #999; text-align: center;">
                红色区域表示异常概率较高，蓝色区域表示正常
            </div>
        </div>
    `;

    modal.onclick = closeHeatmapModal;
    document.body.appendChild(modal);
    document.body.style.overflow = 'hidden';
}

// 关闭热力图模态框
function closeHeatmapModal() {
    const modal = document.getElementById('heatmapModal');
    if (modal) {
        modal.remove();
        document.body.style.overflow = '';
    }
}

// ==================== 音频播放器管理器 ====================

let currentAudio = null;
let currentPlayingButton = null;

// SVG 图标
const playIconSvg = `<svg width="12" height="12" viewBox="0 0 24 24" fill="white"><polygon points="5,3 19,12 5,21"/></svg>`;
const pauseIconSvg = `<svg width="12" height="12" viewBox="0 0 24 24" fill="white"><rect x="5" y="3" width="5" height="18"/><rect x="14" y="3" width="5" height="18"/></svg>`;

function toggleAudioPlay(audioPath, buttonElement) {
    // 如果点击的是当前正在播放的按钮
    if (currentPlayingButton === buttonElement && currentAudio) {
        if (currentAudio.paused) {
            currentAudio.play();
            buttonElement.innerHTML = pauseIconSvg;
            buttonElement.style.background = '#4f46e5';
            buttonElement.title = '点击暂停';
        } else {
            currentAudio.pause();
            buttonElement.innerHTML = playIconSvg;
            buttonElement.style.background = '#6366f1';
            buttonElement.title = '点击试听';
        }
        return;
    }

    // 停止之前播放的音频
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    if (currentPlayingButton) {
        currentPlayingButton.innerHTML = playIconSvg;
        currentPlayingButton.style.background = '#6366f1';
        currentPlayingButton.title = '点击试听';
    }

    // 创建新的音频对象
    currentAudio = new Audio(`${API_BASE}/api/detection/audio/${audioPath}`);
    currentPlayingButton = buttonElement;

    // 更新按钮状态
    buttonElement.innerHTML = pauseIconSvg;
    buttonElement.style.background = '#4f46e5';
    buttonElement.title = '点击暂停';

    // 播放音频
    currentAudio.play().catch(err => {
        console.error('播放失败:', err);
        buttonElement.innerHTML = playIconSvg;
        buttonElement.style.background = '#6366f1';
        buttonElement.title = '点击试听';
        alert('音频播放失败，请检查文件是否存在');
    });

    // 音频结束时的处理
    currentAudio.onended = function() {
        buttonElement.innerHTML = playIconSvg;
        buttonElement.style.background = '#6366f1';
        buttonElement.title = '点击试听';
        currentAudio = null;
        currentPlayingButton = null;
    };

    // 音频错误处理
    currentAudio.onerror = function() {
        console.error('音频加载失败:', audioPath);
        buttonElement.innerHTML = playIconSvg;
        buttonElement.style.background = '#6366f1';
        buttonElement.title = '点击试听';
        currentAudio = null;
        currentPlayingButton = null;
    };
}

// ==================== 页面初始化 ====================

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', async function() {
    console.log('[Init] DOM加载完成，开始初始化...');

    // 加载算法选项（包括自训练模型）
    loadAlgorithmOptions();

    // 初始化离线检测文件上传
    await initOfflineUpload();

    // 初始化特征聚类文件上传
    initClusterUpload();

    // 初始化参考音频库文件上传
    initReferenceUpload();

    // 初始化零样本检测
    initZeroShot();

    // 初始化少样本检测
    initFewShot();

    // 初始化客户端监控
    initClientMonitor();

    console.log('[Init] 所有模块初始化完成');
});

// 动态加载算法选项（包括自训练模型）
async function loadAlgorithmOptions() {
    try {
        const response = await fetch(`${API_BASE}/api/detection/algorithms`);
        const data = await response.json();
        const algorithms = data.algorithms || [];

        // 所有算法 select 元素 ID
        const selectIds = ['algorithm', 'monitorAlgorithm', 'clientAlgorithm'];

        selectIds.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (!select) return;

            const currentValue = select.value;
            select.innerHTML = '';

            // 分组：内置模型
            const builtins = algorithms.filter(a => a.source === 'builtin');
            const customs = algorithms.filter(a => a.source === 'custom');

            if (builtins.length > 0) {
                const group = document.createElement('optgroup');
                group.label = '预置模型';
                builtins.forEach(algo => {
                    const opt = document.createElement('option');
                    opt.value = algo.id;
                    opt.textContent = algo.name;
                    group.appendChild(opt);
                });
                select.appendChild(group);
            }

            if (customs.length > 0) {
                const group = document.createElement('optgroup');
                group.label = '自训练模型';
                customs.forEach(algo => {
                    const opt = document.createElement('option');
                    opt.value = algo.id;
                    opt.textContent = algo.name + ` (${algo.size_mb || '?'}MB)`;
                    group.appendChild(opt);
                });
                select.appendChild(group);
            }

            // 恢复选中值
            if (currentValue) select.value = currentValue;
        });
    } catch (err) {
        console.error('[Init] 加载算法列表失败:', err);
    }
}
