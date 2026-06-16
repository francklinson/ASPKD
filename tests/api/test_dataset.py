"""
数据集构建 API 接口测试
"""
import os
import pytest
import shutil
from pathlib import Path


class TestDatasetReferences:
    """参考音频列表接口测试"""

    @pytest.mark.api
    async def test_get_references(self, client, auth_headers):
        """验证获取参考音频列表"""
        resp = await client.get("/api/dataset/references", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestDatasetStats:
    """数据集统计接口测试"""

    @pytest.mark.smoke
    @pytest.mark.api
    async def test_get_dataset_stats(self, client, auth_headers):
        """验证获取数据集统计信息"""
        resp = await client.get("/api/dataset/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_categories" in data
        assert "total_audio_files" in data
        assert "train_total" in data
        assert "test_total" in data
        assert "categories" in data
        assert isinstance(data["categories"], list)

    @pytest.mark.api
    async def test_stats_values_are_nonnegative(self, client, auth_headers):
        """验证统计值非负"""
        resp = await client.get("/api/dataset/stats", headers=auth_headers)
        data = resp.json()
        assert data["total_categories"] >= 0
        assert data["total_audio_files"] >= 0
        assert data["train_total"] >= 0
        assert data["test_total"] >= 0

    @pytest.mark.api
    async def test_get_category_stats_nonexistent(self, client, auth_headers):
        """验证获取不存在的类别统计"""
        resp = await client.get("/api/dataset/stats/nonexistent_category", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 0


class TestDatasetSegments:
    """音频片段接口测试"""

    @pytest.mark.api
    async def test_get_all_segments(self, client, auth_headers):
        """验证获取所有音频片段"""
        resp = await client.get("/api/dataset/segments", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.api
    async def test_get_segments_with_label_filter(self, client, auth_headers):
        """验证按标签过滤片段"""
        for label in ["normal", "anomaly", "unlabeled"]:
            resp = await client.get(f"/api/dataset/segments?label_filter={label}", headers=auth_headers)
            assert resp.status_code == 200

    @pytest.mark.api
    async def test_get_segments_by_reference(self, client, auth_headers):
        """验证按参考音频获取片段"""
        resp = await client.get("/api/dataset/segments/test_ref", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestDatasetUpload:
    """数据集上传接口测试"""

    @pytest.mark.api
    async def test_upload_manual_invalid_format(self, client, auth_headers):
        """验证上传不支持的文件格式"""
        import io
        files = {"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")}
        data = {"username": "admin"}
        resp = await client.post("/api/dataset/upload-manual", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 400

    @pytest.mark.api
    async def test_upload_manual_wav(self, client, auth_headers, test_wav_file):
        """验证手动上传 WAV 文件"""
        with open(test_wav_file, 'rb') as f:
            files = {"file": ("test_audio.wav", f, "audio/wav")}
            data = {"username": "admin"}
            resp = await client.post("/api/dataset/upload-manual", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 200
        result = resp.json()
        assert result["success"] is True
        assert len(result["segments"]) >= 1

    @pytest.mark.api
    async def test_upload_and_split_invalid_format(self, client, auth_headers):
        """验证上传不支持格式到切分接口"""
        import io
        files = {"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")}
        data = {"username": "admin"}
        resp = await client.post("/api/dataset/upload-and-split", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 400


class TestDatasetAnnotate:
    """数据集标注接口测试"""

    @pytest.mark.api
    async def test_annotate_with_invalid_label(self, client, auth_headers):
        """验证无效标签标注失败"""
        data = {
            "segment_id": "test_seg",
            "label": "invalid_label",
            "category": "test_category",
            "segment_path": "/nonexistent/path.wav"
        }
        resp = await client.post("/api/dataset/annotate", data=data, headers=auth_headers)
        assert resp.status_code == 400
        assert "标签必须是" in resp.json()["detail"]

    @pytest.mark.api
    async def test_annotate_nonexistent_file(self, client, auth_headers):
        """验证标注不存在的文件返回 404"""
        data = {
            "segment_id": "test_seg",
            "label": "normal",
            "category": "test_category",
            "segment_path": "/nonexistent/path.wav"
        }
        resp = await client.post("/api/dataset/annotate", data=data, headers=auth_headers)
        assert resp.status_code == 404


class TestDatasetSplitLog:
    """数据集划分日志接口测试"""

    @pytest.mark.api
    async def test_get_split_logs(self, client, auth_headers):
        """验证获取划分日志"""
        resp = await client.get("/api/dataset/split-log", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.api
    async def test_get_split_logs_with_limit(self, client, auth_headers):
        """验证限制返回日志条数"""
        resp = await client.get("/api/dataset/split-log?limit=5", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) <= 5


class TestDatasetCleanup:
    """数据集清理接口测试"""

    @pytest.mark.api
    async def test_cleanup_temp_files(self, client, auth_headers):
        """验证清理临时文件"""
        resp = await client.post("/api/dataset/cleanup-temp", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["success"] is True
