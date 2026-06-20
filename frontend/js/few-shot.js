// ==================== 少样本检测模块 (已隐藏) ====================

        // ==================== 少样本检测功能 ====================
        let fewShotRefFiles = [];
        let fewShotTestFiles = [];
        let currentFewShotTaskId = null;
        let fewShotRawResults = null;

        // 少样本检测页面初始化
        function initFewShot() {
            // 参考样本上传区
            const refDropZone = document.getElementById('fewShotRefDropZone');
            const refFileInput = document.getElementById('fewShotRefFiles');

            refDropZone.addEventListener('click', () => refFileInput.click());
            refFileInput.addEventListener('change', handleFewShotRefFileSelect);

            refDropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                refDropZone.style.borderColor = '#667eea';
                refDropZone.style.background = '#f0f4ff';
            });
            refDropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                refDropZone.style.borderColor = '#ddd';
                refDropZone.style.background = 'white';
            });
            refDropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                refDropZone.style.borderColor = '#ddd';
                refDropZone.style.background = 'white';
                const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
                if (files.length > 0) updateFewShotRefFileList(files);
            });

            // 测试样本上传区
            const testDropZone = document.getElementById('fewShotTestDropZone');
            const testFileInput = document.getElementById('fewShotTestFiles');

            testDropZone.addEventListener('click', () => testFileInput.click());
            testFileInput.addEventListener('change', handleFewShotTestFileSelect);

            testDropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                testDropZone.style.borderColor = '#667eea';
                testDropZone.style.background = '#f0f4ff';
            });
            testDropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                testDropZone.style.borderColor = '#ddd';
                testDropZone.style.background = 'white';
            });
            testDropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                testDropZone.style.borderColor = '#ddd';
                testDropZone.style.background = 'white';
                const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
                if (files.length > 0) updateFewShotTestFileList(files);
            });
        }

        function handleFewShotRefFileSelect(e) {
            const files = Array.from(e.target.files);
            updateFewShotRefFileList(files);
        }

        function handleFewShotTestFileSelect(e) {
            const files = Array.from(e.target.files);
            updateFewShotTestFileList(files);
        }

        function updateFewShotRefFileList(newFiles) {
            fewShotRefFiles = [...fewShotRefFiles, ...newFiles];
            const fileList = document.getElementById('fewShotRefFileList');
            const dropzone = document.getElementById('fewShotRefDropZone');
            const fileActions = document.getElementById('fewShotRefFileActions');

            if (fewShotRefFiles.length === 0) {
                fileList.innerHTML = '';
                dropzone.style.display = 'block';
                fileActions.style.display = 'none';
            } else {
                dropzone.style.display = 'none';
                fileActions.style.display = 'block';
                fileList.innerHTML = fewShotRefFiles.map((f, index) => `
                    <div class="file-item">
                        <span>📷 ${f.name}</span>
                        <span>${(f.size / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                `).join('');
            }
            updateFewShotAnalyzeButton();
        }

        function updateFewShotTestFileList(newFiles) {
            fewShotTestFiles = [...fewShotTestFiles, ...newFiles];
            const fileList = document.getElementById('fewShotTestFileList');
            const dropzone = document.getElementById('fewShotTestDropZone');
            const fileActions = document.getElementById('fewShotTestFileActions');

            if (fewShotTestFiles.length === 0) {
                fileList.innerHTML = '';
                dropzone.style.display = 'block';
                fileActions.style.display = 'none';
            } else {
                dropzone.style.display = 'none';
                fileActions.style.display = 'block';
                fileList.innerHTML = fewShotTestFiles.map((f, index) => `
                    <div class="file-item">
                        <span>📷 ${f.name}</span>
                        <span>${(f.size / 1024 / 1024).toFixed(2)} MB</span>
                    </div>
                `).join('');
            }
            updateFewShotAnalyzeButton();
        }

        function updateFewShotAnalyzeButton() {
            const analyzeBtn = document.getElementById('analyzeFewShotBtn');
            // 需要至少1个参考样本和1个测试样本
            analyzeBtn.disabled = fewShotRefFiles.length < 1 || fewShotTestFiles.length < 1;
        }

        function clearFewShotRefFiles() {
            fewShotRefFiles = [];
            document.getElementById('fewShotRefFiles').value = '';
            updateFewShotRefFileList([]);
        }

        function clearFewShotTestFiles() {
            fewShotTestFiles = [];
            document.getElementById('fewShotTestFiles').value = '';
            updateFewShotTestFileList([]);
        }

        function continueFewShotRefUpload() {
            document.getElementById('fewShotRefFiles').value = '';
            document.getElementById('fewShotRefFiles').click();
        }

        function continueFewShotTestUpload() {
            document.getElementById('fewShotTestFiles').value = '';
            document.getElementById('fewShotTestFiles').click();
        }

        async function analyzeFewShot() {
            const btn = document.getElementById('analyzeFewShotBtn');
            const progress = document.getElementById('fewShotProgress');
            const progressFill = document.getElementById('fewShotProgressFill');
            const progressText = document.getElementById('fewShotProgressText');
            const resultCard = document.getElementById('fewShotResultCard');
            const resultsDiv = document.getElementById('fewShotResults');

            if (fewShotRefFiles.length === 0) {
                await showModal('请先上传至少一个参考样本（正常图像）', '提示');
                return;
            }
            if (fewShotTestFiles.length === 0) {
                await showModal('请先上传至少一个测试样本（待检测图像）', '提示');
                return;
            }

            try {
                btn.disabled = true;
                btn.textContent = '⏳ 分析中...';
                progress.style.display = 'block';
                progressFill.style.width = '10%';
                progressText.textContent = '正在上传...';

                resultCard.style.display = 'none';
                resultsDiv.innerHTML = '<p style="text-align: center; color: #666; margin: 40px 0;">正在分析中，请稍候...</p>';

                const formData = new FormData();
                fewShotRefFiles.forEach(file => formData.append('reference_files', file));
                fewShotTestFiles.forEach(file => formData.append('test_files', file));
                formData.append('backbone', document.getElementById('fewShotBackbone').value);
                formData.append('threshold', document.getElementById('fewShotThreshold').value);
                formData.append('k_shot', document.getElementById('fewShotK').value);
                formData.append('pca_ev', document.getElementById('fewShotPcaEv').value);
                formData.append('score_method', document.getElementById('fewShotScoreMethod').value);

                const response = await fetch(`${API_BASE}/api/few-shot/analyze`, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.task_id) {
                    currentFewShotTaskId = result.task_id;
                    progressFill.style.width = '50%';
                    progressText.textContent = '正在分析中，请稍候...';
                    pollFewShotResult(currentFewShotTaskId);
                } else {
                    throw new Error(result.detail || '创建分析任务失败');
                }
            } catch (error) {
                console.error('少样本检测失败:', error);
                await showModal('分析失败: ' + error.message, '错误');
                btn.disabled = false;
                btn.textContent = '🚀 开始少样本检测';
                progress.style.display = 'none';
            }
        }

        async function pollFewShotResult(taskId) {
            const progress = document.getElementById('fewShotProgress');
            const progressFill = document.getElementById('fewShotProgressFill');
            const progressText = document.getElementById('fewShotProgressText');
            const btn = document.getElementById('analyzeFewShotBtn');

            const checkResult = async () => {
                try {
                    const response = await fetch(`${API_BASE}/api/few-shot/result/${taskId}`);
                    const result = await response.json();

                    if (result.status === 'completed') {
                        progressFill.style.width = '100%';
                        progressText.textContent = '分析完成!';
                        displayFewShotResult(result);
                        btn.disabled = false;
                        btn.textContent = '🚀 开始少样本检测';
                        progress.style.display = 'none';
                    } else if (result.status === 'failed') {
                        throw new Error(result.error || '分析失败');
                    } else {
                        progressFill.style.width = '70%';
                        progressText.textContent = '正在分析中...';
                        setTimeout(checkResult, 2000);
                    }
                } catch (error) {
                    console.error('获取结果失败:', error);
                    await showModal('获取结果失败: ' + error.message, '错误');
                    btn.disabled = false;
                    btn.textContent = '🚀 开始少样本检测';
                    progress.style.display = 'none';
                }
            };
            checkResult();
        }

        function displayFewShotResult(result) {
            fewShotRawResults = result;
            const resultCard = document.getElementById('fewShotResultCard');
            resultCard.style.display = 'block';
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            renderFewShotResults();
        }

        function renderFewShotResults() {
            if (!fewShotRawResults) {
                console.log('[FewShot] renderFewShotResults: fewShotRawResults is null');
                return;
            }

            const resultsDiv = document.getElementById('fewShotResults');
            const reportDiv = document.getElementById('fewShotReport');
            const threshold = parseFloat(document.getElementById('fewShotThreshold').value);
            const result = fewShotRawResults;

            console.log('[FewShot] renderFewShotResults: threshold=', threshold, 'results count=', result.results.length);

            const resultsWithThreshold = result.results.map(r => {
                const is_anomaly = r.anomaly_score >= threshold;
                console.log(`[FewShot] ${r.filename}: score=${r.anomaly_score.toFixed(4)}, threshold=${threshold}, is_anomaly=${is_anomaly}`);
                return {
                    ...r,
                    is_anomaly: is_anomaly
                };
            });

            const anomalyCount = resultsWithThreshold.filter(r => r.is_anomaly).length;
            const normalCount = resultsWithThreshold.length - anomalyCount;

            let summaryHtml = `
                <div style="background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
                    <h3 style="margin: 0 0 12px 0; font-size: 16px;">📊 检测摘要</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; font-size: 14px;">
                        <div><strong>总文件数:</strong> ${result.summary.total_files}</div>
                        <div><strong>异常数:</strong> <span style="color: #dc3545;">${anomalyCount}</span></div>
                        <div><strong>正常数:</strong> <span style="color: #28a745;">${normalCount}</span></div>
                        <div><strong>当前阈值:</strong> <span style="color: #667eea;">${threshold.toFixed(2)}</span></div>
                        <div><strong>参考样本数:</strong> ${result.summary.k_shot}</div>
                        <div><strong>骨干网络:</strong> ${result.summary.backbone}</div>
                    </div>
                </div>
            `;

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
                                    <div style="font-size: 11px; color: #666; margin-bottom: 4px;">🔥 异常热力图:</div>
                                    <img src="${API_BASE}${r.overlay_url || r.heatmap_url}"
                                         style="width: 100%; height: 150px; object-fit: contain; background: #f5f5f5; border-radius: 4px; cursor: pointer;"
                                         onclick="showHeatmapModal('${API_BASE}${r.original_url}', '${API_BASE}${r.overlay_url}', '${API_BASE}${r.heatmap_url}', '${r.filename}', ${r.anomaly_score}, ${isAnomaly})"
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

            if (result.report) {
                reportDiv.style.display = 'block';
                reportDiv.textContent = result.report;
            }
        }

        function updateFewShotDisplay() {
            console.log('[FewShot] updateFewShotDisplay called, fewShotRawResults:', fewShotRawResults);
            if (fewShotRawResults) {
                const threshold = parseFloat(document.getElementById('fewShotThreshold').value);
                console.log('[FewShot] Threshold changed to:', threshold);
                renderFewShotResults();
            } else {
                console.log('[FewShot] fewShotRawResults is null, skipping render');
            }
        }

        function clearFewShotResult() {
            const resultCard = document.getElementById('fewShotResultCard');
            const resultsDiv = document.getElementById('fewShotResults');
            const reportDiv = document.getElementById('fewShotReport');

            resultCard.style.display = 'none';
            resultsDiv.innerHTML = '';
            reportDiv.style.display = 'none';
            reportDiv.textContent = '';
            fewShotRawResults = null;
            currentFewShotTaskId = null;
        }
