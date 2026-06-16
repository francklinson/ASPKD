#!/bin/bash
# 测试运行脚本
# 用法:
#   bash tests/run_tests.sh              # 运行所有测试
#   bash tests/run_tests.sh smoke        # 运行冒烟测试
#   bash tests/run_tests.sh api          # 运行接口测试
#   bash tests/run_tests.sh regression   # 运行回归测试
#   bash tests/run_tests.sh ui           # 运行UI测试（需要先启动服务）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 确保测试目录存在
mkdir -p tests/test_data

# 检查 pytest 和相关依赖
echo "============================================"
echo "  音频异常检测系统 - 自动化测试套件"
echo "============================================"
echo ""

check_deps() {
    python -c "import pytest" 2>/dev/null || {
        echo "[INFO] 安装 pytest 和相关依赖..."
        pip install pytest pytest-asyncio pytest-timeout httpx -q
    }
}

run_smoke() {
    echo "[SMOKE] 运行冒烟测试..."
    pytest tests/smoke/ -v -m smoke --tb=short --timeout=120 2>&1
}

run_api() {
    echo "[API] 运行接口测试..."
    pytest tests/api/ -v -m "api" --tb=short --timeout=120 2>&1
}

run_regression() {
    echo "[REGRESSION] 运行回归测试..."
    pytest tests/regression/ -v -m regression --tb=short --timeout=300 2>&1
}

run_ui() {
    echo "[UI] 运行 UI 测试..."
    # 检查 Playwright
    python -c "from playwright.async_api import async_playwright" 2>/dev/null || {
        echo "[INFO] 安装 Playwright..."
        pip install playwright -q
        playwright install chromium
    }
    # 检查服务是否运行
    if ! curl -s http://localhost:8004/health > /dev/null 2>&1; then
        echo "[WARN] 后端服务未运行，请先启动: bash start_server.sh 或 python backend/main.py"
        echo "[INFO] 跳过需要服务器的 UI 测试"
    fi
    pytest tests/ui/ -v -m ui --tb=short --timeout=60 2>&1
}

run_all() {
    echo "[ALL] 运行全部测试套件..."
    echo ""
    run_smoke
    echo ""
    run_api
    echo ""
    run_regression
    echo ""
    run_ui
}

# 检查依赖
check_deps

case "${1:-all}" in
    smoke)
        run_smoke
        ;;
    api)
        run_api
        ;;
    regression)
        run_regression
        ;;
    ui)
        run_ui
        ;;
    all)
        run_all
        ;;
    *)
        echo "用法: $0 {smoke|api|regression|ui|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  测试完成"
echo "============================================"
