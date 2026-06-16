"""
冒烟测试 - 快速验证核心功能可用

运行方式:
    pytest tests/smoke/ -v -m smoke
    pytest tests/smoke/ -v --timeout=60
"""
import pytest
import asyncio


class TestSmokeHealthCheck:
    """冒烟测试: 服务健康检查"""

    @pytest.mark.smoke
    async def test_server_is_alive(self, client):
        """验证服务启动且响应正常"""
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestSmokeAuth:
    """冒烟测试: 用户认证"""

    @pytest.mark.smoke
    async def test_login_flow(self, client):
        """验证完整的登录流程"""
        # 1. 登录
        resp = await client.post("/api/auth/login", data={
            "username": "admin", "password": "tp123456"
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        token = resp.json()["token"]

        # 2. 验证会话
        headers = {"Authorization": f"Bearer {token}"}
        resp = await client.get("/api/auth/session", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"

        # 3. 登出
        resp = await client.post("/api/auth/logout", headers=headers)
        assert resp.status_code == 200

        # 4. 验证登出后会话失效
        resp = await client.get("/api/auth/session", headers=headers)
        assert resp.status_code == 401


class TestSmokeAPI:
    """冒烟测试: 核心 API 可用性"""

    @pytest.mark.smoke
    async def test_algorithms_endpoint(self, client, auth_headers):
        """验证算法列表 API 可用"""
        resp = await client.get("/api/detection/algorithms", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["algorithms"]) >= 4

    @pytest.mark.smoke
    async def test_devices_endpoint(self, client, auth_headers):
        """验证设备列表 API 可用"""
        resp = await client.get("/api/detection/devices", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["devices"]) >= 2  # auto + cpu

    @pytest.mark.smoke
    async def test_tasks_endpoint(self, client, auth_headers):
        """验证任务列表 API 可用"""
        resp = await client.get("/api/tasks/list", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert "total" in data

    @pytest.mark.smoke
    async def test_dataset_stats_endpoint(self, client, auth_headers):
        """验证数据集统计 API 可用"""
        resp = await client.get("/api/dataset/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total_categories" in data


class TestSmokePages:
    """冒烟测试: 前端页面可访问性"""

    @pytest.mark.smoke
    async def test_login_page(self, client):
        """验证登录页面可访问"""
        resp = await client.get("/login")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.smoke
    async def test_main_page(self, client):
        """验证主页面可访问"""
        resp = await client.get("/main")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.smoke
    async def test_dataset_page(self, client):
        """验证数据集页面可访问"""
        resp = await client.get("/dataset")
        assert resp.status_code == 200

    @pytest.mark.smoke
    async def test_training_page(self, client):
        """验证训练页面可访问"""
        resp = await client.get("/training")
        assert resp.status_code == 200

    @pytest.mark.smoke
    async def test_api_docs(self, client):
        """验证 API 文档可访问"""
        resp = await client.get("/docs")
        assert resp.status_code == 200


class TestSmokeConcurrency:
    """冒烟测试: 并发请求"""

    @pytest.mark.smoke
    async def test_concurrent_health_checks(self, client):
        """验证并发健康检查请求"""
        tasks = [client.get("/health") for _ in range(10)]
        results = await asyncio.gather(*tasks)
        for resp in results:
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"

    @pytest.mark.smoke
    async def test_concurrent_login_requests(self, client):
        """验证并发登录请求"""
        async def login_user(username):
            resp = await client.post("/api/auth/login", data={
                "username": username, "password": "tp123456"
            })
            return resp

        tasks = [login_user(f"user{i}") for i in range(1, 4)]
        results = await asyncio.gather(*tasks)
        for resp in results:
            assert resp.status_code == 200
            assert resp.json()["success"] is True
