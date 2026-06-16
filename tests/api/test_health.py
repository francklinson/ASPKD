"""
健康检查 & 基础路由接口测试
"""
import pytest


class TestHealthCheck:
    """健康检查接口测试"""

    @pytest.mark.smoke
    @pytest.mark.api
    async def test_health_returns_healthy(self, client):
        """验证 /health 返回 healthy 状态"""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "audio-anomaly-detection-api"
        assert data["version"] == "2.0.0"

    @pytest.mark.api
    async def test_root_returns_login_page(self, client):
        """验证 / 根路径返回登录页面"""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.api
    async def test_main_page_accessible(self, client):
        """验证 /main 页面可访问"""
        resp = await client.get("/main")
        assert resp.status_code == 200

    @pytest.mark.api
    async def test_login_page_accessible(self, client):
        """验证 /login 页面可访问"""
        resp = await client.get("/login")
        assert resp.status_code == 200

    @pytest.mark.api
    async def test_dataset_page_accessible(self, client):
        """验证 /dataset 页面可访问"""
        resp = await client.get("/dataset")
        assert resp.status_code == 200

    @pytest.mark.api
    async def test_training_page_accessible(self, client):
        """验证 /training 页面可访问"""
        resp = await client.get("/training")
        assert resp.status_code == 200

    @pytest.mark.api
    async def test_favicon_accessible(self, client):
        """验证 favicon 可访问"""
        resp = await client.get("/favicon.ico")
        assert resp.status_code in (200, 404)

    @pytest.mark.api
    async def test_api_docs_accessible(self, client):
        """验证 API 文档可访问"""
        resp = await client.get("/docs")
        assert resp.status_code == 200
