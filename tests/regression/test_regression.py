"""
回归测试 - 端到端业务流程验证

确保关键业务流程端到端正常工作

运行方式:
    pytest tests/regression/ -v -m regression
"""
import os
import io
import pytest
import asyncio
from pathlib import Path


class TestRegressionAuthFlow:
    """回归测试: 用户认证完整流程"""

    @pytest.mark.regression
    async def test_complete_auth_lifecycle(self, client):
        """验证完整的认证生命周期"""
        # 1. 未认证访问需要保护的接口
        resp = await client.get("/api/auth/session")
        assert resp.status_code == 401

        # 2. 登录
        resp = await client.post("/api/auth/login", data={
            "username": "admin", "password": "tp123456"
        })
        assert resp.status_code == 200
        token = resp.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 3. 认证后访问
        resp = await client.get("/api/auth/session", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"

        # 4. 使用无效 token 访问
        resp = await client.get("/api/auth/session", headers={
            "Authorization": "Bearer invalid_token_123"
        })
        assert resp.status_code == 401

        # 5. 登出
        resp = await client.post("/api/auth/logout", headers=headers)
        assert resp.status_code == 200

        # 6. 登出后原 token 失效
        resp = await client.get("/api/auth/session", headers=headers)
        assert resp.status_code == 401

    @pytest.mark.regression
    async def test_multiple_user_sessions(self, client):
        """验证多用户会话隔离"""
        # 用户1登录
        resp1 = await client.post("/api/auth/login", data={
            "username": "user1", "password": "tp123456"
        })
        token1 = resp1.json()["token"]
        headers1 = {"Authorization": f"Bearer {token1}"}

        # 用户2登录
        resp2 = await client.post("/api/auth/login", data={
            "username": "user2", "password": "tp123456"
        })
        token2 = resp2.json()["token"]
        headers2 = {"Authorization": f"Bearer {token2}"}

        # 验证各自的会话
        session1 = await client.get("/api/auth/session", headers=headers1)
        session2 = await client.get("/api/auth/session", headers=headers2)
        assert session1.json()["username"] == "user1"
        assert session2.json()["username"] == "user2"

        # 用户1登出，用户2仍然有效
        await client.post("/api/auth/logout", headers=headers1)
        resp = await client.get("/api/auth/session", headers=headers1)
        assert resp.status_code == 401
        resp = await client.get("/api/auth/session", headers=headers2)
        assert resp.status_code == 200


class TestRegressionDetectionFlow:
    """回归测试: 检测任务流程"""

    @pytest.mark.regression
    async def test_detection_workflow(self, client, auth_headers, test_wav_file):
        """验证完整的检测任务流程"""
        # 1. 获取可用算法
        resp = await client.get("/api/detection/algorithms", headers=auth_headers)
        assert resp.status_code == 200
        algorithms = resp.json()["algorithms"]
        assert len(algorithms) > 0

        # 2. 获取设备列表
        resp = await client.get("/api/detection/devices", headers=auth_headers)
        assert resp.status_code == 200
        devices = resp.json()["devices"]
        assert len(devices) >= 2

        # 3. 上传文件创建检测任务
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
        task_id = result["task_id"]

        # 4. 查询任务结果
        resp = await client.get(f"/api/detection/result/{task_id}", headers=auth_headers)
        assert resp.status_code == 200
        task_result = resp.json()
        assert task_result["task_id"] == task_id

        # 5. 任务在任务列表中可见
        resp = await client.get("/api/tasks/list", headers=auth_headers)
        tasks = resp.json()["tasks"]
        task_ids = [t["id"] for t in tasks]
        assert task_id in task_ids

    @pytest.mark.regression
    async def test_task_list_filtering(self, client, auth_headers):
        """验证任务列表过滤功能"""
        # 按状态过滤
        for status in ["pending", "running", "completed", "failed"]:
            resp = await client.get(f"/api/tasks/list?status={status}", headers=auth_headers)
            assert resp.status_code == 200

        # 分页
        resp = await client.get("/api/tasks/list?limit=5&offset=0", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["tasks"]) <= 5


class TestRegressionDatasetFlow:
    """回归测试: 数据集构建流程"""

    @pytest.mark.regression
    async def test_dataset_workflow(self, client, auth_headers, test_wav_file):
        """验证完整的数据集操作流程"""
        # 1. 查看数据集统计
        resp = await client.get("/api/dataset/stats", headers=auth_headers)
        assert resp.status_code == 200
        initial_stats = resp.json()

        # 2. 上传音频文件
        with open(test_wav_file, 'rb') as f:
            files = {"file": ("test_audio.wav", f, "audio/wav")}
            data = {"username": "admin"}
            resp = await client.post("/api/dataset/upload-manual", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 200
        upload_result = resp.json()
        assert upload_result["success"] is True

        # 3. 查看音频片段
        resp = await client.get("/api/dataset/segments", headers=auth_headers)
        assert resp.status_code == 200

        # 4. 查看划分日志
        resp = await client.get("/api/dataset/split-log", headers=auth_headers)
        assert resp.status_code == 200

        # 5. 清理临时文件
        resp = await client.post("/api/dataset/cleanup-temp", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestRegressionTrainingFlow:
    """回归测试: 训练流程"""

    @pytest.mark.regression
    async def test_training_workflow(self, client, auth_headers):
        """验证训练相关接口"""
        # 1. 获取数据集
        resp = await client.get("/api/training/datasets", headers=auth_headers)
        assert resp.status_code == 200
        datasets = resp.json()

        # 2. 获取已训练模型
        resp = await client.get("/api/training/models", headers=auth_headers)
        assert resp.status_code == 200
        models = resp.json()

        # 3. 查看训练任务列表
        resp = await client.get("/api/training/status", headers=auth_headers)
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)


class TestRegressionZeroShotFlow:
    """回归测试: 零样本检测流程"""

    @pytest.mark.regression
    async def test_zero_shot_workflow(self, client, auth_headers, test_image_file):
        """验证零样本检测完整流程"""
        # 1. 获取可用的 backbone
        resp = await client.get("/api/zero-shot/backbones", headers=auth_headers)
        assert resp.status_code == 200
        backbones = resp.json()["backbones"]
        assert len(backbones) > 0

        # 2. 上传文件进行分析
        import io
        with open(test_image_file, 'rb') as f:
            content = f.read()

        files = []
        for i in range(5):
            files.append(("files", (f"test_{i}.png", io.BytesIO(content), "image/png")))

        data = {
            "backbone": "musc_clip_l14_336",
            "threshold": "0.5",
            "batch_size": "4",
            "r_list": "1,3,5"
        }
        resp = await client.post("/api/zero-shot/analyze", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "queued"
        assert "task_id" in result


class TestRegressionFewShotFlow:
    """回归测试: 少样本检测流程"""

    @pytest.mark.regression
    async def test_few_shot_workflow(self, client, auth_headers, test_image_file):
        """验证少样本检测完整流程"""
        # 1. 获取可用的 backbone
        resp = await client.get("/api/few-shot/backbones", headers=auth_headers)
        assert resp.status_code == 200
        backbones = resp.json()["backbones"]
        assert len(backbones) > 0

        # 2. 上传文件进行分析
        import io
        with open(test_image_file, 'rb') as f:
            content = f.read()

        files = [
            ("reference_files", ("ref_0.png", io.BytesIO(content), "image/png")),
            ("test_files", ("test_0.png", io.BytesIO(content), "image/png")),
        ]
        data = {
            "backbone": "subspacead_dinov2_large_672",
            "threshold": "0.5",
            "k_shot": "1",
            "pca_ev": "0.99",
            "score_method": "reconstruction"
        }
        resp = await client.post("/api/few-shot/analyze", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "queued"
        assert "task_id" in result


class TestRegressionTaskCleanup:
    """回归测试: 任务清理流程"""

    @pytest.mark.regression
    async def test_task_cleanup_workflow(self, client, auth_headers):
        """验证任务清理完整流程"""
        # 1. 获取当前任务统计
        resp = await client.get("/api/tasks/stats", headers=auth_headers)
        assert resp.status_code == 200

        # 2. 执行清理
        resp = await client.post(
            "/api/tasks/cleanup?keep_days=30&include_files=true",
            headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleaned"

        # 3. 清理后的任务列表仍然可访问
        resp = await client.get("/api/tasks/list", headers=auth_headers)
        assert resp.status_code == 200
