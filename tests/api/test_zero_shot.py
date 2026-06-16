"""
Zero-Shot 异常检测 API 接口测试
"""
import pytest


class TestZeroShotBackbones:
    """Backbone 列表接口测试"""

    @pytest.mark.api
    async def test_get_backbones(self, client, auth_headers):
        """验证获取 backbone 列表"""
        resp = await client.get("/api/zero-shot/backbones", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "backbones" in data
        assert isinstance(data["backbones"], list)
        assert len(data["backbones"]) > 0

    @pytest.mark.api
    async def test_backbone_structure(self, client, auth_headers):
        """验证每个 backbone 包含必要字段"""
        resp = await client.get("/api/zero-shot/backbones", headers=auth_headers)
        backbones = resp.json()["backbones"]
        for bb in backbones:
            assert "id" in bb
            assert "name" in bb


class TestZeroShotAnalyze:
    """零样本分析接口测试"""

    @pytest.mark.api
    async def test_analyze_without_files_returns_error(self, client, auth_headers):
        """验证不上传文件时返回错误"""
        resp = await client.post("/api/zero-shot/analyze", headers=auth_headers)
        assert resp.status_code in (400, 422)

    @pytest.mark.api
    async def test_analyze_insufficient_files(self, client, auth_headers, test_image_file):
        """验证文件少于5个时返回错误"""
        import io
        files = []
        with open(test_image_file, 'rb') as f:
            content = f.read()
        for i in range(3):  # 少于5个
            files.append(("files", (f"test_{i}.png", io.BytesIO(content), "image/png")))

        data = {"backbone": "musc_clip_l14_336", "threshold": "0.5"}
        resp = await client.post("/api/zero-shot/analyze", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 400
        assert "至少需要" in resp.json()["detail"]

    @pytest.mark.api
    async def test_analyze_invalid_backbone(self, client, auth_headers, test_image_file):
        """验证无效 backbone 返回错误"""
        import io
        files = []
        with open(test_image_file, 'rb') as f:
            content = f.read()
        for i in range(5):
            files.append(("files", (f"test_{i}.png", io.BytesIO(content), "image/png")))

        data = {"backbone": "invalid_backbone_name", "threshold": "0.5"}
        resp = await client.post("/api/zero-shot/analyze", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 400

    @pytest.mark.api
    async def test_analyze_invalid_r_list(self, client, auth_headers, test_image_file):
        """验证无效的 r_list 参数"""
        import io
        files = []
        with open(test_image_file, 'rb') as f:
            content = f.read()
        for i in range(5):
            files.append(("files", (f"test_{i}.png", io.BytesIO(content), "image/png")))

        data = {"backbone": "musc_clip_l14_336", "r_list": "invalid"}
        resp = await client.post("/api/zero-shot/analyze", files=files, data=data, headers=auth_headers)
        # 应该能正常处理（使用默认值）
        assert resp.status_code == 200


class TestZeroShotResult:
    """零样本结果查询接口测试"""

    @pytest.mark.api
    async def test_get_nonexistent_result_404(self, client, auth_headers):
        """验证查询不存在的结果返回 404"""
        resp = await client.get("/api/zero-shot/result/nonexistent-task-id", headers=auth_headers)
        assert resp.status_code == 404
