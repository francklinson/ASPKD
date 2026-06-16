"""
UI 测试公共 Fixtures (Playwright)
"""
import os
import pytest
import pytest_asyncio
from pathlib import Path

# 检查 Playwright 是否安装
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# 后端服务地址（需要先启动服务）
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8004")


@pytest_asyncio.fixture
async def browser():
    """创建 Playwright 浏览器实例"""
    if not PLAYWRIGHT_AVAILABLE:
        pytest.skip("Playwright 未安装，跳过 UI 测试。安装: pip install playwright && playwright install chromium")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest_asyncio.fixture
async def page(browser):
    """创建新页面"""
    context = await browser.new_context(
        viewport={"width": 1280, "height": 720},
        ignore_https_errors=True
    )
    page = await context.new_page()
    yield page
    await context.close()


def is_server_running():
    """检查测试服务器是否在运行"""
    import socket
    host = "localhost"
    port = 8004
    try:
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError):
        return False


@pytest.fixture(scope="session")
def server_running():
    """验证服务器是否在运行（session 级别）"""
    if not is_server_running():
        pytest.skip(
            "后端服务未运行。请先启动服务: python backend/main.py 或 bash start_server.sh"
        )
    return True
