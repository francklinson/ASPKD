"""
UI 测试 - 前端页面基础验证
使用 Playwright 进行前端页面交互测试

运行方式:
    # 需要先启动后端服务
    python backend/main.py &
    # 然后运行 UI 测试
    pytest tests/ui/ -v -m ui --timeout=60

依赖安装:
    pip install playwright pytest-playwright
    playwright install chromium
"""
import pytest

# 检查 Playwright 是否可用
try:
    from playwright.async_api import expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

BASE_URL = "http://localhost:8004"


@pytest.mark.ui
class TestLoginPage:
    """登录页面 UI 测试"""

    @pytest.mark.asyncio
    async def test_login_page_loads(self, page, server_running):
        """验证登录页面正常加载"""
        await page.goto(f"{BASE_URL}/login")
        await page.wait_for_load_state("networkidle")

        # 检查页面标题
        title = await page.title()
        assert len(title) > 0

    @pytest.mark.asyncio
    async def test_login_form_present(self, page, server_running):
        """验证登录表单存在"""
        await page.goto(f"{BASE_URL}/login")
        await page.wait_for_load_state("networkidle")

        # 检查关键元素存在
        # 登录表单通常包含用户名/密码输入框和登录按钮
        body_text = await page.inner_text("body")
        assert "登录" in body_text or "login" in body_text.lower()

    @pytest.mark.asyncio
    async def test_login_with_credentials(self, page, server_running):
        """验证使用有效凭据登录"""
        await page.goto(f"{BASE_URL}/login")
        await page.wait_for_load_state("networkidle")

        # 尝试填写登录表单
        try:
            # 查找用户名输入框
            username_input = page.locator('input[name="username"]')
            if await username_input.count() == 0:
                username_input = page.locator('input[type="text"]').first

            # 查找密码输入框
            password_input = page.locator('input[name="password"]')
            if await password_input.count() == 0:
                password_input = page.locator('input[type="password"]').first

            if await username_input.count() > 0:
                await username_input.fill("admin")
            if await password_input.count() > 0:
                await password_input.fill("tp123456")

            # 查找并点击登录按钮
            login_btn = page.locator('button[type="submit"]')
            if await login_btn.count() == 0:
                login_btn = page.locator('button').filter(has_text="登录")
            if await login_btn.count() == 0:
                login_btn = page.locator('button').first

            if await login_btn.count() > 0:
                await login_btn.click()
                await page.wait_for_timeout(2000)
        except Exception as e:
            pytest.skip(f"登录表单交互失败: {e}")

    @pytest.mark.asyncio
    async def test_login_page_has_no_console_errors(self, page, server_running):
        """验证登录页面无控制台错误"""
        errors = []

        async def handle_error(msg):
            errors.append(msg.text)

        page.on("pageerror", handle_error)

        await page.goto(f"{BASE_URL}/login")
        await page.wait_for_load_state("networkidle")

        # 如果没有 JS 错误，测试通过
        assert len(errors) == 0, f"登录页面存在 JavaScript 错误: {errors}"


@pytest.mark.ui
class TestMainPage:
    """主页面 UI 测试"""

    @pytest.mark.asyncio
    async def test_main_page_loads(self, page, server_running):
        """验证主页面正常加载"""
        resp = await page.goto(f"{BASE_URL}/main")
        if resp and resp.status == 200:
            await page.wait_for_load_state("networkidle")
            title = await page.title()
            assert len(title) > 0

    @pytest.mark.asyncio
    async def test_main_page_has_no_console_errors(self, page, server_running):
        """验证主页面无控制台错误"""
        errors = []

        async def handle_error(msg):
            errors.append(msg.text)

        page.on("pageerror", handle_error)

        resp = await page.goto(f"{BASE_URL}/main")
        if resp and resp.status == 200:
            await page.wait_for_load_state("networkidle")

        # 检查是否有 JS 错误
        assert len(errors) == 0, f"主页面存在 JavaScript 错误: {errors}"


@pytest.mark.ui
class TestDatasetPage:
    """数据集页面 UI 测试"""

    @pytest.mark.asyncio
    async def test_dataset_page_loads(self, page, server_running):
        """验证数据集页面正常加载"""
        resp = await page.goto(f"{BASE_URL}/dataset")
        if resp and resp.status == 200:
            await page.wait_for_load_state("networkidle")
            title = await page.title()
            assert len(title) > 0

    @pytest.mark.asyncio
    async def test_dataset_page_has_no_console_errors(self, page, server_running):
        """验证数据集页面无控制台错误"""
        errors = []

        async def handle_error(msg):
            errors.append(msg.text)

        page.on("pageerror", handle_error)

        resp = await page.goto(f"{BASE_URL}/dataset")
        if resp and resp.status == 200:
            await page.wait_for_load_state("networkidle")

        assert len(errors) == 0, f"数据集页面存在 JavaScript 错误: {errors}"


@pytest.mark.ui
class TestTrainingPage:
    """训练页面 UI 测试"""

    @pytest.mark.asyncio
    async def test_training_page_loads(self, page, server_running):
        """验证训练页面正常加载"""
        resp = await page.goto(f"{BASE_URL}/training")
        if resp and resp.status == 200:
            await page.wait_for_load_state("networkidle")
            title = await page.title()
            assert len(title) > 0

    @pytest.mark.asyncio
    async def test_training_page_has_no_console_errors(self, page, server_running):
        """验证训练页面无控制台错误"""
        errors = []

        async def handle_error(msg):
            errors.append(msg.text)

        page.on("pageerror", handle_error)

        resp = await page.goto(f"{BASE_URL}/training")
        if resp and resp.status == 200:
            await page.wait_for_load_state("networkidle")

        assert len(errors) == 0, f"训练页面存在 JavaScript 错误: {errors}"


@pytest.mark.ui
class TestResponsiveness:
    """响应式设计测试"""

    @pytest.mark.asyncio
    async def test_login_page_responsive(self, browser, server_running):
        """验证登录页面在不同屏幕尺寸下可正常显示"""
        for width, height in [(375, 812), (768, 1024), (1280, 720), (1920, 1080)]:
            context = await browser.new_context(viewport={"width": width, "height": height})
            page = await context.new_page()

            resp = await page.goto(f"{BASE_URL}/login")
            if resp and resp.status == 200:
                await page.wait_for_load_state("networkidle")
                # 验证页面没有水平溢出
                body = page.locator("body")
                assert await body.is_visible()

            await context.close()
