// ==================== 零样本检测模块 (已隐藏) ====================

        // ==================== 零样本检测功能 ====================
        let zeroShotFiles = [];
        let currentZeroShotTaskId = null;
        let zeroShotRawResults = null;  // 存储原始检测结果，用于阈值调整时重新渲染

        // 零样本检测页面初始化
        function initZeroShot() {
            const dropZone = document.getElementById('zeroShotDropZone');
            const fileInput = document.getElementById('zeroShotFiles');

            // 点击拖拽区域选择文件
            dropZone.addEventListener('click', () => fileInput.click());

            // 文件选择
            fileInput.addEventListener('change', handleZeroShotFileSelect);

            // 拖拽事件
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = '#667eea';
                dropZone.style.background = '#f0f4ff';
            });

            dropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = '#ddd';
                dropZone.style.background = 'white';
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = '#ddd';
                dropZone.style.background = 'white';
                const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
                if (files.length > 0) {
                    updateZeroShotFileList(files);
                }
            });

        }

        function handleZeroShotFileSelect(e) {
            const files = Array.from(e.target.files);
            updateZeroShotFileList(files);
        }

        function updateZeroShotFileList(newFiles) {
            zeroShotFiles = [...zeroShotFiles, ...newFiles];
            const fileList = document.getElementById('zeroShotFileList');
            const dropzone = document.getElementById('zeroShotDropZone');
            const fileActions = document.getElementById('zeroShotFileActions');
            const analyzeBtn = document.getElementById('analyzeZeroShotBtn');

            if (zeroShotFiles.length === 0) {
                fileList.innerHTML = '';
                dropzone.style.display = 'block';
                fileActions.style.display = 'none';
                analyzeBtn.disabled = false;
                return;
            }

            dropzone.style.display = 'none';
            fileActions.style.display = 'block';
            // 少于5个文件时禁用分析按钮
            analyzeBtn.disabled = zeroShotFiles.length < 5;

            fileList.innerHTML = zeroShotFiles.map((f, index) => `
                <div class="file-item">
                    <span>📷 ${f.name}</span>
                    <span>${(f.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
            `).join('');
        }

        function clearZeroShotFiles() {
            zeroShotFiles = [];
            document.getElementById('zeroShotFiles').value = '';
            updateZeroShotFileList([]);
        }

        function continueZeroShotUpload() {
            document.getElementById('zeroShotFiles').value = '';
            document.getElementById('zeroShotFiles').click();
        }

        async function analyzeZeroShot() {
            const btn = document.getElementById('analyzeZeroShotBtn');
            const progress = document.getElementById('zeroShotProgress');
            const progressFill = document.getElementById('zeroShotProgressFill');
            const progressText = document.getElementById('zeroShotProgressText');
            const resultCard = document.getElementById('zeroShotResultCard');
            const resultsDiv = document.getElementById('zeroShotResults');

            if (zeroShotFiles.length === 0) {
                await showModal('请先上传至少一个图像文件', '提示');
                return;
            }

            // MuSc 算法需要至少5个样本作为参考集
            if (zeroShotFiles.length < 5) {
                await showModal('MuSc 零样本检测需要至少 5 个图像文件才能进行有效分析。<br><br>当前文件数: ' + zeroShotFiles.length + '<br><br>请上传更多图像文件以继续。', '提示');
                return;
            }

            try {
                btn.disabled = true;
                btn.textContent = '⏳ 分析中...';
                progress.style.display = 'block';
                progressFill.style.width = '10%';
                progressText.textContent = '正在上传...';

                // 隐藏之前的结果
                resultCard.style.display = 'none';
                resultsDiv.innerHTML = '<p style="text-align: center; color: #666; margin: 40px 0;">正在分析中，请稍候...</p>';

                const formData = new FormData();
                zeroShotFiles.forEach(file => formData.append('files', file));
                formData.append('backbone', document.getElementById('zeroShotBackbone').value);
                formData.append('threshold', document.getElementById('zeroShotThreshold').value);
                formData.append('batch_size', document.getElementById('zeroShotBatchSize').value);
                formData.append('r_list', document.getElementById('zeroShotRList').value);

                const response = await fetch(`${API_BASE}/api/zero-shot/analyze`, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.task_id) {
                    currentZeroShotTaskId = result.task_id;
                    progressFill.style.width = '50%';
                    progressText.textContent = '正在分析中，请稍候...';

                    // 开始轮询结果
                    pollZeroShotResult(currentZeroShotTaskId);
                } else {
                    throw new Error(result.detail || '创建分析任务失败');
                }
            } catch (error) {
                console.error('零样本检测失败:', error);
                await showModal('分析失败: ' + error.message, '错误');

                btn.disabled = false;
                btn.textContent = '🚀 开始零样本检测';
                progress.style.display = 'none';
            }
        }

        async function pollZeroShotResult(taskId) {
            const progress = document.getElementById('zeroShotProgress');
            const progressFill = document.getElementById('zeroShotProgressFill');
            const progressText = document.getElementById('zeroShotProgressText');
            const btn = document.getElementById('analyzeZeroShotBtn');

            const checkResult = async () => {
                try {
                    const response = await fetch(`${API_BASE}/api/zero-shot/result/${taskId}`);
                    const result = await response.json();

                    if (result.status === 'completed') {
                        // 分析完成
                        progressFill.style.width = '100%';
                        progressText.textContent = '分析完成!';

                        displayZeroShotResult(result);

                        btn.disabled = false;
                        btn.textContent = '🚀 开始零样本检测';
                        progress.style.display = 'none';

                    } else if (result.status === 'failed') {
                        // 分析失败
                        throw new Error(result.error || '分析失败');
                    } else {
                        // 仍在处理中
                        progressFill.style.width = '70%';
                        progressText.textContent = '正在分析中...';
                        setTimeout(checkResult, 2000);
                    }
                } catch (error) {
                    console.error('获取结果失败:', error);
                    await showModal('获取结果失败: ' + error.message, '错误');

                    btn.disabled = false;
                    btn.textContent = '🚀 开始零样本检测';
                    progress.style.display = 'none';
                }
            };

            checkResult();
        }

        function displayZeroShotResult(result) {
            // 保存原始结果数据，用于阈值调整时重新渲染
            zeroShotRawResults = result;

            const resultCard = document.getElementById('zeroShotResultCard');

            // 显示结果卡片
            resultCard.style.display = 'block';

            // 滚动到结果区域
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

            // 调试信息
            console.log('[ZeroShot] 显示结果:', result);
            if (result.results && result.results.length > 0) {
                console.log('[ZeroShot] 第一个结果:', result.results[0]);
                console.log('[ZeroShot] heatmap_url:', result.results[0].heatmap_url);
                console.log('[ZeroShot] overlay_url:', result.results[0].overlay_url);
            }

            // 根据当前阈值渲染结果
            renderZeroShotResults();
        }

        function renderZeroShotResults() {
            if (!zeroShotRawResults) return;

            const resultsDiv = document.getElementById('zeroShotResults');
            const reportDiv = document.getElementById('zeroShotReport');
            const threshold = parseFloat(document.getElementById('zeroShotThreshold').value);
            const result = zeroShotRawResults;

            // 根据当前阈值重新计算异常状态
            const resultsWithThreshold = result.results.map(r => ({
                ...r,
                is_anomaly: r.anomaly_score >= threshold
            }));

            // 重新统计
            const anomalyCount = resultsWithThreshold.filter(r => r.is_anomaly).length;
            const normalCount = resultsWithThreshold.length - anomalyCount;

            // 显示摘要（使用动态计算的统计）
            let summaryHtml = `
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                    <h3 style="margin: 0 0 12px 0; font-size: 16px;">📊 检测摘要</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 14px;">
                        <div><strong>总文件数:</strong> ${result.summary.total_files}</div>
                        <div><strong>异常数:</strong> <span style="color: #dc3545;">${anomalyCount}</span></div>
                        <div><strong>正常数:</strong> <span style="color: #28a745;">${normalCount}</span></div>
                        <div><strong>当前阈值:</strong> <span style="color: #667eea;">${threshold.toFixed(2)}</span></div>
                        <div><strong>平均异常分数:</strong> ${result.summary.avg_anomaly_score.toFixed(4)}</div>
                        <div><strong>平均推理时间:</strong> ${result.summary.avg_inference_time.toFixed(2)} ms</div>
                        <div><strong>骨干网络:</strong> ${result.summary.backbone}</div>
                    </div>
                </div>
            `;

            // 显示详细结果和热力图
            let resultsHtml = `
                <h4 style="margin: 16px 0 12px 0; font-size: 15px;">📋 详细结果</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px;">
            `;

            resultsWithThreshold.forEach((r, index) => {
                const isAnomaly = r.is_anomaly;
                const statusColor = isAnomaly ? '#dc3545' : '#28a745';
                const statusBg = isAnomaly ? '#f8d7da' : '#d4edda';
                const hasHeatmap = r.heatmap_url !== null && r.heatmap_url !== undefined;
                const hasOverlay = r.overlay_url !== null && r.overlay_url !== undefined;
                const displayUrl = hasOverlay ? r.overlay_url : r.heatmap_url;

                resultsHtml += `
                    <div style="border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: white;">
                        <div style="padding: 12px; border-bottom: 1px solid #e0e0e0; background: #f8f9fa;">
                            <div style="font-weight: 600; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${r.filename}">
                                📷 ${r.filename}
                            </div>
                        </div>
                        <div style="padding: 12px;">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; font-size: 12px;">
                                <div><strong>异常分数:</strong> ${r.anomaly_score.toFixed(4)}</div>
                                <div><strong>状态:</strong>
                                    <span style="background: ${statusBg}; color: ${statusColor}; padding: 2px 6px; border-radius: 4px; font-size: 11px;">
                                        ${isAnomaly ? '🔴 异常' : '🟢 正常'}
                                    </span>
                                </div>
                                <div><strong>推理时间:</strong> ${r.inference_time.toFixed(2)}ms</div>
                                <div><strong>序号:</strong> #${index + 1}</div>
                            </div>
                            ${hasHeatmap ? `
                                <div style="margin-top: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <div style="font-size: 11px; color: #666;">🔥 异常热力图:</div>
                                        <div style="display: flex; gap: 4px;">
                                            <button id="btn-original-${index}"
                                                    onclick="toggleHeatmapMode(${index}, 'original')"
                                                    style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">原图</button>
                                            <button id="btn-overlay-${index}"
                                                    onclick="toggleHeatmapMode(${index}, 'overlay')"
                                                    style="padding: 2px 8px; font-size: 10px; border: 1px solid #667eea; background: #667eea; color: white; border-radius: 4px; cursor: pointer;">叠加</button>
                                            <button id="btn-heatmap-${index}"
                                                    onclick="toggleHeatmapMode(${index}, 'heatmap')"
                                                    style="padding: 2px 8px; font-size: 10px; border: 1px solid #d9d9d9; background: white; color: #666; border-radius: 4px; cursor: pointer;">纯热力图</button>
                                        </div>
                                    </div>
                                    <img id="heatmap-img-${index}"
                                         src="${API_BASE}${displayUrl}"
                                         data-original="${r.original_url ? API_BASE + r.original_url : ''}"
                                         data-overlay="${API_BASE}${r.overlay_url || r.heatmap_url}"
                                         data-heatmap="${API_BASE}${r.heatmap_url}"
                                         style="width: 100%; border-radius: 4px; border: 1px solid #e0e0e0; cursor: pointer;"
                                         onclick="showHeatmapModalFromImg('heatmap-img-${index}', '${r.filename}', ${r.anomaly_score})"
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

            resultsHtml += `
                </div>
            `;

            resultsDiv.innerHTML = summaryHtml + resultsHtml;

            // 显示报告
            if (result.report) {
                reportDiv.style.display = 'block';
                reportDiv.textContent = result.report;
            }
        }

        function updateZeroShotDisplay() {
            // 阈值滑动条变化时，重新渲染结果
            if (zeroShotRawResults) {
                renderZeroShotResults();
            }
        }

        function clearZeroShotResult() {
            const resultCard = document.getElementById('zeroShotResultCard');
            const resultsDiv = document.getElementById('zeroShotResults');
            const reportDiv = document.getElementById('zeroShotReport');

            resultCard.style.display = 'none';
            resultsDiv.innerHTML = '';
            reportDiv.style.display = 'none';
            reportDiv.textContent = '';

            // 清除保存的原始结果数据
            zeroShotRawResults = null;

            // 清除当前任务ID
            currentZeroShotTaskId = null;
        }

        // 切换热力图显示模式
        function toggleHeatmapMode(index, mode) {
            const img = document.getElementById(`heatmap-img-${index}`);
            const btnOriginal = document.getElementById(`btn-original-${index}`);
            const btnOverlay = document.getElementById(`btn-overlay-${index}`);
            const btnHeatmap = document.getElementById(`btn-heatmap-${index}`);

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

        // 从图片元素获取URL显示热力图大图
        function showHeatmapModalFromImg(imgId, filename, score) {
            const img = document.getElementById(imgId);
            if (img) {
                showHeatmapModal(img.src, filename, score);
            }
        }
