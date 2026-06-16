"""
任务管理 API 接口测试
"""
import pytest


class TestTaskList:
    """任务列表接口测试"""

    @pytest.mark.smoke
    @pytest.mark.api
    async def test_list_tasks_returns_valid_response(self, client, auth_headers):
        """验证获取任务列表返回正确结构"""
        resp = await client.get("/api/tasks/list", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert "total" in data
        assert "running" in data
        assert "queued" in data
        assert isinstance(data["tasks"], list)

    @pytest.mark.api
    async def test_list_tasks_with_status_filter(self, client, auth_headers):
        """验证按状态过滤任务列表"""
        for status in ["pending", "running", "completed", "failed", "cancelled"]:
            resp = await client.get(f"/api/tasks/list?status={status}", headers=auth_headers)
            assert resp.status_code == 200

    @pytest.mark.api
    async def test_list_tasks_with_pagination(self, client, auth_headers):
        """验证任务列表分页"""
        # 测试不同的 limit 和 offset
        resp = await client.get("/api/tasks/list?limit=5&offset=0", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["tasks"]) <= 5

        resp = await client.get("/api/tasks/list?limit=10&offset=10", headers=auth_headers)
        assert resp.status_code == 200

    @pytest.mark.api
    async def test_list_tasks_default_params(self, client, auth_headers):
        """验证默认查询参数"""
        resp = await client.get("/api/tasks/list", headers=auth_headers)
        data = resp.json()
        # 默认 limit=50, offset=0
        assert len(data["tasks"]) <= 50


class TestTaskStats:
    """任务统计接口测试"""

    @pytest.mark.api
    async def test_get_task_stats(self, client, auth_headers):
        """验证获取任务统计信息"""
        resp = await client.get("/api/tasks/stats", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.api
    async def test_task_count_consistent(self, client, auth_headers):
        """验证任务列表中的 total 与列表长度一致"""
        resp = await client.get("/api/tasks/list", headers=auth_headers)
        data = resp.json()
        assert data["total"] >= len(data["tasks"])


class TestTaskCleanup:
    """任务清理接口测试"""

    @pytest.mark.api
    async def test_cleanup_old_tasks(self, client, auth_headers):
        """验证清理旧任务"""
        resp = await client.post(
            "/api/tasks/cleanup?keep_days=7&include_files=true",
            headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cleaned"
        assert "removed_count" in data
        assert "file_stats" in data

    @pytest.mark.api
    async def test_cleanup_with_clear_all(self, client, auth_headers):
        """验证清理全部已完成任务"""
        resp = await client.post(
            "/api/tasks/cleanup?clear_all=true&include_files=false",
            headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleaned"

    @pytest.mark.api
    async def test_cleanup_default_params(self, client, auth_headers):
        """验证默认清理参数"""
        resp = await client.post("/api/tasks/cleanup", headers=auth_headers)
        assert resp.status_code == 200


class TestTaskDelete:
    """任务删除接口测试"""

    @pytest.mark.api
    async def test_delete_nonexistent_task_404(self, client, auth_headers):
        """验证删除不存在的任务返回 404"""
        resp = await client.delete("/api/tasks/nonexistent-task-id", headers=auth_headers)
        assert resp.status_code == 404
