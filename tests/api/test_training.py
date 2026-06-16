"""
模型训练 API 接口测试
"""
import pytest


class TestTrainingDatasets:
    """训练数据集接口测试"""

    @pytest.mark.api
    async def test_get_training_datasets(self, client, auth_headers):
        """验证获取训练数据集列表"""
        resp = await client.get("/api/training/datasets", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.api
    async def test_dataset_structure(self, client, auth_headers):
        """验证每个数据集包含必要字段"""
        resp = await client.get("/api/training/datasets", headers=auth_headers)
        datasets = resp.json()
        for ds in datasets:
            assert "name" in ds
            assert "train_normal_count" in ds
            assert "test_normal_count" in ds
            assert "test_anomaly_count" in ds
            assert "total_count" in ds
            assert "trainable" in ds

    @pytest.mark.api
    async def test_get_dataset_stats_nonexistent(self, client, auth_headers):
        """验证获取不存在的类别统计"""
        resp = await client.get("/api/training/dataset-stats/nonexistent_category", headers=auth_headers)
        assert resp.status_code == 404


class TestTrainedModels:
    """已训练模型接口测试"""

    @pytest.mark.api
    async def test_get_trained_models(self, client, auth_headers):
        """验证获取已训练模型列表"""
        resp = await client.get("/api/training/models", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.api
    async def test_get_model_detail_nonexistent(self, client, auth_headers):
        """验证获取不存在的模型详情"""
        resp = await client.get("/api/training/models/nonexistent.pth", headers=auth_headers)
        assert resp.status_code == 404


class TestTrainingTasks:
    """训练任务接口测试"""

    @pytest.mark.api
    async def test_list_training_tasks(self, client, auth_headers):
        """验证列出所有训练任务"""
        resp = await client.get("/api/training/status", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.api
    async def test_get_training_status_nonexistent(self, client, auth_headers):
        """验证获取不存在的训练任务状态"""
        resp = await client.get("/api/training/status/nonexistent_task", headers=auth_headers)
        assert resp.status_code == 404

    @pytest.mark.api
    async def test_start_training_invalid_category(self, client, auth_headers):
        """验证使用无效类别启动训练"""
        data = {
            "categories": ["nonexistent_category"],
            "model_type": "dinov3",
            "model_size": "small",
            "total_iters": 1000,
            "batch_size": 8
        }
        resp = await client.post("/api/training/start", json=data, headers=auth_headers)
        assert resp.status_code == 400

    @pytest.mark.api
    async def test_start_training_empty_categories(self, client, auth_headers):
        """验证空类别列表启动训练"""
        data = {
            "categories": [],
            "model_type": "dinov3",
            "model_size": "small",
            "total_iters": 1000,
            "batch_size": 8
        }
        resp = await client.post("/api/training/start", json=data, headers=auth_headers)
        assert resp.status_code in (400, 422)

    @pytest.mark.api
    async def test_stop_training_nonexistent(self, client, auth_headers):
        """验证停止不存在的训练任务"""
        resp = await client.post("/api/training/stop/nonexistent_task", headers=auth_headers)
        assert resp.status_code == 404
