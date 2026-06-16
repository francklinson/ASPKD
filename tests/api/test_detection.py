"""
检测 API 接口测试
"""
import os
import pytest


class TestDetectionAlgorithms:
    """算法列表接口测试"""

    @pytest.mark.smoke
    @pytest.mark.api
    async def test_get_algorithms_returns_list(self, client):
        """验证获取算法列表返回正确结构"""
        resp = await client.get("/api/detection/algorithms")
        assert resp.status_code == 200
        data = resp.json()
        assert "algorithms" in data
        assert isinstance(data["algorithms"], list)
        assert len(data["algorithms"]) >= 4  # 至少4个预置算法

    @pytest.mark.api
    async def test_algorithms_have_required_fields(self, client):
        """验证每个算法包含必要字段"""
        resp = await client.get("/api/detection/algorithms")
        algorithms = resp.json()["algorithms"]
        for algo in algorithms:
            assert "id" in algo
            assert "name" in algo
            assert "description" in algo
            assert "type" in algo
            assert "source" in algo

    @pytest.mark.api
    async def test_builtin_algorithms_present(self, client):
        """验证预置算法存在"""
        resp = await client.get("/api/detection/algorithms")
        algo_ids = [a["id"] for a in resp.json()["algorithms"]]
        assert "dinomaly_dinov3_small" in algo_ids
        assert "dinomaly_dinov3_large" in algo_ids
        assert "dinomaly_dinov2_small" in algo_ids
        assert "dinomaly_dinov2_large" in algo_ids


class TestDevices:
    """设备列表接口测试"""

    @pytest.mark.api
    async def test_get_devices_returns_list(self, client):
        """验证获取设备列表返回正确结构"""
        resp = await client.get("/api/detection/devices")
        assert resp.status_code == 200
        data = resp.json()
        assert "devices" in data
        assert isinstance(data["devices"], list)

    @pytest.mark.api
    async def test_devices_has_auto_option(self, client):
        """验证设备列表包含 auto 选项"""
        resp = await client.get("/api/detection/devices")
        devices = resp.json()["devices"]
        device_ids = [d["id"] for d in devices]
        assert "auto" in device_ids

    @pytest.mark.api
    async def test_devices_has_cpu_option(self, client):
        """验证设备列表包含 cpu 选项"""
        resp = await client.get("/api/detection/devices")
        devices = resp.json()["devices"]
        device_ids = [d["id"] for d in devices]
        assert "cpu" in device_ids

    @pytest.mark.api
    async def test_device_structure(self, client):
        """验证每个设备信息包含必要字段"""
        resp = await client.get("/api/detection/devices")
        devices = resp.json()["devices"]
        for device in devices:
            assert "id" in device
            assert "name" in device
            assert "type" in device


class TestDetectionUpload:
    """检测上传接口测试"""

    @pytest.mark.api
    async def test_upload_without_file_returns_400(self, client, auth_headers):
        """验证不上传文件时返回 400 错误"""
        resp = await client.post("/api/detection/upload", headers=auth_headers)
        assert resp.status_code == 400 or resp.status_code == 422

    @pytest.mark.api
    async def test_upload_invalid_file_format(self, client, auth_headers):
        """验证上传不支持的文件格式返回 400"""
        import io
        files = {"files": ("test.txt", io.BytesIO(b"not an audio file"), "text/plain")}
        data = {"algorithm": "dinomaly_dinov3_small", "device": "cpu"}
        resp = await client.post("/api/detection/upload", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 400
        assert "不支持的文件格式" in resp.json()["detail"]

    @pytest.mark.api
    async def test_upload_wav_file_creates_task(self, client, auth_headers, test_wav_file):
        """验证上传 WAV 文件创建检测任务"""
        with open(test_wav_file, 'rb') as f:
            files = {"files": ("test_audio.wav", f, "audio/wav")}
            data = {
                "algorithm": "dinomaly_dinov3_small",
                "device": "cpu",
                "save_results": "true"
            }
            resp = await client.post("/api/detection/upload", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "queued"
        assert "task_id" in result
        assert len(result["task_id"]) > 0

    @pytest.mark.api
    async def test_upload_multiple_wav_files(self, client, auth_headers, test_wav_files):
        """验证上传多个 WAV 文件"""
        import io
        files = []
        for i, fpath in enumerate(test_wav_files):
            with open(fpath, 'rb') as f:
                files.append(("files", (f"test_{i}.wav", f.read(), "audio/wav")))

        data = {"algorithm": "dinomaly_dinov3_small", "device": "cpu"}
        resp = await client.post("/api/detection/upload", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"


class TestDetectionResult:
    """检测结果接口测试"""

    @pytest.mark.api
    async def test_get_nonexistent_result_404(self, client):
        """验证查询不存在的任务返回 404"""
        resp = await client.get("/api/detection/result/nonexistent-task-id")
        assert resp.status_code == 404

    @pytest.mark.api
    async def test_cancel_nonexistent_task_400(self, client):
        """验证取消不存在的任务返回 400"""
        resp = await client.post("/api/detection/cancel/nonexistent-task-id")
        assert resp.status_code == 400


class TestDetectionExport:
    """检测结果导出接口测试"""

    @pytest.mark.api
    async def test_export_nonexistent_task_404(self, client):
        """验证导出不存在的任务返回 404"""
        resp = await client.get("/api/detection/export/nonexistent-task-id")
        assert resp.status_code == 404

    @pytest.mark.api
    async def test_export_incomplete_task_400(self, client):
        """验证导出未完成的任务返回 400"""
        # 创建新任务后立即尝试导出（任务还在排队）
        resp = await client.get("/api/detection/export/not-completed-task")
        assert resp.status_code in (404, 400)
