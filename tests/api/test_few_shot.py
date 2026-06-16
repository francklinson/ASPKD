"""
Few-Shot (少样本) 异常检测 API 接口测试
"""
import pytest


class TestFewShotBackbones:
    """Backbone 列表接口测试"""

    @pytest.mark.api
    async def test_get_backbones(self, client, auth_headers):
        """验证获取 backbone 列表"""
        resp = await client.get("/api/few-shot/backbones", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "backbones" in data
        assert isinstance(data["backbones"], list)
        assert len(data["backbones"]) > 0


class TestFewShotAnalyze:
    """少样本分析接口测试"""

    @pytest.mark.api
    async def test_analyze_without_reference_files(self, client, auth_headers):
        """验证不上传参考文件时返回错误"""
        import io
        test_files = [("test_files", ("test.png", io.BytesIO(b"fake"), "image/png"))]
        resp = await client.post("/api/few-shot/analyze", files=test_files, headers=auth_headers)
        assert resp.status_code in (400, 422)

    @pytest.mark.api
    async def test_analyze_without_test_files(self, client, auth_headers, test_image_file):
        """验证只上传参考文件不上传测试文件时返回错误"""
        import io
        with open(test_image_file, 'rb') as f:
            content = f.read()
        ref_files = [("reference_files", ("ref.png", io.BytesIO(content), "image/png"))]
        resp = await client.post("/api/few-shot/analyze", files=ref_files, headers=auth_headers)
        assert resp.status_code in (400, 422)

    @pytest.mark.api
    async def test_analyze_invalid_backbone(self, client, auth_headers, test_image_file):
        """验证无效 backbone 返回错误"""
        import io
        with open(test_image_file, 'rb') as f:
            content = f.read()
        files = [
            ("reference_files", ("ref.png", io.BytesIO(content), "image/png")),
            ("test_files", ("test.png", io.BytesIO(content), "image/png")),
        ]
        data = {"backbone": "invalid_backbone_name"}
        resp = await client.post("/api/few-shot/analyze", files=files, data=data, headers=auth_headers)
        assert resp.status_code == 400

    @pytest.mark.api
    async def test_analyze_with_params(self, client, auth_headers, test_image_file):
        """验证带参数的少样本分析请求"""
        import io
        with open(test_image_file, 'rb') as f:
            content = f.read()
        files = [
            ("reference_files", ("ref.png", io.BytesIO(content), "image/png")),
            ("test_files", ("test.png", io.BytesIO(content), "image/png")),
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


class TestFewShotResult:
    """少样本结果查询接口测试"""

    @pytest.mark.api
    async def test_get_nonexistent_result_404(self, client, auth_headers):
        """验证查询不存在的结果返回 404"""
        resp = await client.get("/api/few-shot/result/nonexistent-task-id", headers=auth_headers)
        assert resp.status_code == 404
