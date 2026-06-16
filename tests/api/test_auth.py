"""
用户认证 API 接口测试
"""
import pytest


class TestAuthLogin:
    """登录接口测试"""

    @pytest.mark.smoke
    @pytest.mark.api
    async def test_login_success_admin(self, client):
        """验证管理员登录成功"""
        resp = await client.post("/api/auth/login", data={
            "username": "admin",
            "password": "tp123456"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["token"] is not None
        assert data["username"] == "admin"
        assert len(data["token"]) == 32

    @pytest.mark.api
    async def test_login_success_user1(self, client):
        """验证普通用户登录成功"""
        resp = await client.post("/api/auth/login", data={
            "username": "user1",
            "password": "tp123456"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["username"] == "user1"

    @pytest.mark.api
    async def test_login_success_user5(self, client):
        """验证 user5 登录成功"""
        resp = await client.post("/api/auth/login", data={
            "username": "user5",
            "password": "tp123456"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @pytest.mark.api
    async def test_login_invalid_password(self, client):
        """验证错误密码登录失败"""
        resp = await client.post("/api/auth/login", data={
            "username": "admin",
            "password": "wrongpassword"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["token"] is None

    @pytest.mark.api
    async def test_login_invalid_username(self, client):
        """验证不存在的用户登录失败"""
        resp = await client.post("/api/auth/login", data={
            "username": "nonexistent",
            "password": "tp123456"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    @pytest.mark.api
    async def test_login_user6_not_exist(self, client):
        """验证 user6 不在合法用户列表中"""
        resp = await client.post("/api/auth/login", data={
            "username": "user6",
            "password": "tp123456"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    @pytest.mark.api
    async def test_login_sets_token_in_body(self, client):
        """验证登录成功返回的 token 格式正确"""
        resp = await client.post("/api/auth/login", data={
            "username": "admin",
            "password": "tp123456"
        })
        data = resp.json()
        assert len(data["token"]) > 0
        # token 应该是 hex 字符串
        assert all(c in "0123456789abcdef" for c in data["token"].lower())


class TestAuthSession:
    """会话管理接口测试"""

    @pytest.mark.api
    async def test_get_session_with_valid_token(self, client, auth_headers):
        """验证有效 token 可以获取会话信息"""
        resp = await client.get("/api/auth/session", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"
        assert data["token"] is not None

    @pytest.mark.api
    async def test_get_session_without_token(self, client):
        """验证无 token 时获取会话返回 401"""
        resp = await client.get("/api/auth/session")
        assert resp.status_code == 401

    @pytest.mark.api
    async def test_get_session_with_invalid_token(self, client):
        """验证无效 token 返回 401"""
        resp = await client.get("/api/auth/session", headers={
            "Authorization": "Bearer invalidtoken1234567890"
        })
        assert resp.status_code == 401

    @pytest.mark.api
    async def test_logout_clears_session(self, client, auth_headers):
        """验证登出后会话失效"""
        resp = await client.post("/api/auth/logout", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        # 登出后再次请求需要认证的接口应返回 401
        resp2 = await client.get("/api/auth/session", headers=auth_headers)
        assert resp2.status_code == 401

    @pytest.mark.api
    async def test_logout_without_token(self, client):
        """验证无 token 登出不会报错"""
        resp = await client.post("/api/auth/logout")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @pytest.mark.api
    async def test_list_sessions(self, client):
        """验证列出所有活跃会话"""
        resp = await client.get("/api/auth/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "sessions" in data
        assert isinstance(data["sessions"], list)


class TestAuthBoundary:
    """认证边界测试"""

    @pytest.mark.api
    async def test_login_empty_username(self, client):
        """验证空用户名登录"""
        resp = await client.post("/api/auth/login", data={
            "username": "",
            "password": "tp123456"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    @pytest.mark.api
    async def test_login_empty_password(self, client):
        """验证空密码登录"""
        resp = await client.post("/api/auth/login", data={
            "username": "admin",
            "password": ""
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    @pytest.mark.api
    async def test_login_special_chars_username(self, client):
        """验证特殊字符用户名"""
        resp = await client.post("/api/auth/login", data={
            "username": "admin' OR '1'='1",
            "password": "tp123456"
        })
        assert resp.status_code == 200
        data = resp.json()
        # 应该安全处理，返回失败而非被 SQL 注入
        assert data["success"] is False

    @pytest.mark.api
    async def test_multiple_logins_same_user(self, client):
        """验证同一用户多次登录生成不同 token"""
        resp1 = await client.post("/api/auth/login", data={
            "username": "admin", "password": "tp123456"
        })
        resp2 = await client.post("/api/auth/login", data={
            "username": "admin", "password": "tp123456"
        })
        token1 = resp1.json()["token"]
        token2 = resp2.json()["token"]
        assert token1 != token2, "同一用户每次登录应生成不同的 token"
