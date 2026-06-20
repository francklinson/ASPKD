// ==================== 特征聚类模块 (已隐藏) ====================

        // 特征聚类状态
        let clusterSelectedFiles = [];
        let currentClusterTaskId = null;
        let clusterWsConnection = null;

        // ==================== 特征聚类分析功能 ====================

        function initClusterPage() {
            // 初始化特征聚类页面
            console.log('[Cluster] 初始化特征聚类页面');
        }

        // 特征聚类文件上传处理函数
        function handleClusterFiles(files) {
            const allowedExtensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'];
            const newFiles = Array.from(files).filter(file => {
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                return allowedExtensions.includes(ext);
            });

            if (newFiles.length === 0) {
                showModal('请选择有效的音频文件 (WAV, MP3, FLAC等)', '提示');
                return;
            }

            clusterSelectedFiles = [...clusterSelectedFiles, ...newFiles];
            updateClusterFileList();
        }

        function updateClusterFileList() {
            const list = document.getElementById('clusterFileList');
            const dropzone = document.getElementById('clusterDropzone');
            const fileActions = document.getElementById('clusterFileActions');
            const startBtn = document.getElementById('clusterStartBtn');

            if (clusterSelectedFiles.length === 0) {
                list.innerHTML = '';
                dropzone.style.display = 'block';
                fileActions.style.display = 'none';
                startBtn.disabled = true;
                return;
            }

            dropzone.style.display = 'none';
            fileActions.style.display = 'block';
            startBtn.disabled = clusterSelectedFiles.length < 2;

            list.innerHTML = clusterSelectedFiles.map((f, index) => `
                <div class="file-item">
                    <span>📄 ${f.name}</span>
                    <span>${(f.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
            `).join('');
        }

        function clearClusterFiles() {
            clusterSelectedFiles = [];
            document.getElementById('clusterFileInput').value = '';
            updateClusterFileList();
        }

        function continueClusterUpload() {
            document.getElementById('clusterFileInput').value = '';
            document.getElementById('clusterFileInput').click();
        }

        async function startClusterAnalysis() {
            if (clusterSelectedFiles.length < 2) {
                await showModal('请至少上传2个音频文件进行分析', '提示');
                return;
            }

            const btn = document.getElementById('clusterStartBtn');
            const progress = document.getElementById('clusterProgress');
            const progressFill = document.getElementById('clusterProgressFill');
            const progressText = document.getElementById('clusterProgressText');

            btn.disabled = true;
            btn.textContent = '🔬 分析中...';
            progress.style.display = 'block';
            progressFill.style.width = '10%';
            progressText.textContent = '正在上传文件...';

            try {
                const formData = new FormData();
                clusterSelectedFiles.forEach(f => formData.append('files', f));
                formData.append('extractor_type', document.getElementById('clusterExtractor').value);
                formData.append('n_clusters', document.getElementById('clusterNClusters').value);
                formData.append('anomaly_threshold', document.getElementById('clusterThreshold').value);
                formData.append('use_3d', document.getElementById('clusterUse3D').checked);
                formData.append('tsne_perplexity', '30');

                progressFill.style.width = '30%';
                progressText.textContent = '正在创建分析任务...';

                const response = await fetch(`${API_BASE}/api/cluster/analyze`, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    currentClusterTaskId = result.task_id;
                    progressFill.style.width = '50%';
                    progressText.textContent = '正在分析中，请稍候...';

                    // 开始轮询结果
                    pollClusterResult(currentClusterTaskId);
                } else {
                    throw new Error(result.detail || '创建分析任务失败');
                }
            } catch (error) {
                console.error('聚类分析失败:', error);
                await showModal('分析失败: ' + error.message, '错误');

                btn.disabled = false;
                btn.textContent = '🔬 开始聚类分析';
                progress.style.display = 'none';
            }
        }

        async function pollClusterResult(taskId) {
            const progress = document.getElementById('clusterProgress');
            const progressFill = document.getElementById('clusterProgressFill');
            const progressText = document.getElementById('clusterProgressText');
            const btn = document.getElementById('clusterStartBtn');

            const checkResult = async () => {
                try {
                    const response = await fetch(`${API_BASE}/api/cluster/result/${taskId}`);
                    const result = await response.json();

                    if (result.status === 'completed') {
                        // 分析完成
                        progressFill.style.width = '100%';
                        progressText.textContent = '分析完成!';

                        displayClusterResult(result);

                        btn.disabled = false;
                        btn.textContent = '🔬 开始聚类分析';
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
                    await showModal('获取分析结果失败: ' + error.message, '错误');

                    btn.disabled = false;
                    btn.textContent = '🔬 开始聚类分析';
                    progress.style.display = 'none';
                }
            };

            checkResult();
        }

        function displayClusterResult(result) {
            const resultCard = document.getElementById('clusterResultCard');
            const resultImage = document.getElementById('clusterResultImage');
            const resultReport = document.getElementById('clusterReport');
            const interactiveContainer = document.getElementById('clusterInteractiveContainer');
            const interactiveFrame = document.getElementById('clusterInteractiveFrame');

            resultCard.style.display = 'block';

            // 调试日志
            console.log('[Cluster] 显示结果:', result);
            console.log('[Cluster] interactive_html:', result.interactive_html);
            console.log('[Cluster] result_image:', result.result_image);

            // 显示交互式 3D 可视化 (如果可用)
            if (result.interactive_html) {
                const htmlUrl = `${API_BASE}/${result.interactive_html}`;
                console.log('[Cluster] 3D可视化URL:', htmlUrl);
                interactiveFrame.src = htmlUrl;
                interactiveContainer.style.display = 'block';
                // 静态图片作为备选/缩略图
                if (result.result_image) {
                    resultImage.src = `${API_BASE}/${result.result_image}`;
                    resultImage.style.display = 'block';
                }
            } else if (result.result_image) {
                // 只有静态图片
                interactiveContainer.style.display = 'none';
                resultImage.src = `${API_BASE}/${result.result_image}`;
                resultImage.style.display = 'block';
            } else {
                interactiveContainer.style.display = 'none';
                resultImage.style.display = 'none';
            }

            if (result.report) {
                // 简单的Markdown渲染
                let html = result.report
                    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
                    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                    .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
                    .replace(/\n/gim, '<br>');
                resultReport.innerHTML = html;
            } else {
                resultReport.textContent = '暂无详细报告';
            }

            // 滚动到结果区域
            resultCard.scrollIntoView({ behavior: 'smooth' });
        }

        function clearClusterResult() {
            const resultCard = document.getElementById('clusterResultCard');
            const resultImage = document.getElementById('clusterResultImage');
            const resultReport = document.getElementById('clusterReport');
            const interactiveContainer = document.getElementById('clusterInteractiveContainer');
            const interactiveFrame = document.getElementById('clusterInteractiveFrame');

            resultCard.style.display = 'none';
            resultImage.src = '';
            resultReport.innerHTML = '';
            interactiveFrame.src = '';
            interactiveContainer.style.display = 'none';
        }

        // 特征聚类文件上传初始化
        function initClusterUpload() {
            const clusterDropzone = document.getElementById('clusterDropzone');
            const clusterFileInput = document.getElementById('clusterFileInput');

            if (clusterDropzone && clusterFileInput) {
                clusterDropzone.addEventListener('click', () => clusterFileInput.click());

                clusterDropzone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    clusterDropzone.classList.add('dragover');
                });

                clusterDropzone.addEventListener('dragleave', () => {
                    clusterDropzone.classList.remove('dragover');
                });

                clusterDropzone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    clusterDropzone.classList.remove('dragover');
                    handleClusterFiles(e.dataTransfer.files);
                });

                clusterFileInput.addEventListener('change', (e) => {
                    handleClusterFiles(e.target.files);
                });

                console.log('[Cluster] 文件上传事件监听器已绑定');
            } else {
                console.error('[Cluster] 找不到上传区域元素:', { clusterDropzone: !!clusterDropzone, clusterFileInput: !!clusterFileInput });
            }
        }
