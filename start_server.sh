#!/bin/bash
# ASD 音频异常检测服务管理脚本
# 支持: start | stop | restart | status

set -e

# 配置
PROJECT_DIR="/home/zhouchenghao/PycharmProjects/ASD_for_SPK"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PID_FILE="$PROJECT_DIR/.service.pid"
CURRENT_LOG_FILE="$PROJECT_DIR/.current_log"  # 记录当前日志文件路径
HOST="0.0.0.0"
PORT="8004"

# 自动检测 Python 版本
get_python_version() {
    if [ -f "$VENV_PYTHON" ]; then
        "$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12"
    else
        echo "3.12"
    fi
}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助
show_help() {
    echo -e "${BLUE}ASD 音频异常检测服务管理脚本${NC}"
    echo ""
    echo "用法: ./service.sh [命令]"
    echo ""
    echo "命令:"
    echo "  start      启动服务"
    echo "  stop       停止服务"
    echo "  restart    重启服务"
    echo "  status     查看服务状态"
    echo "  log        查看实时日志"
    echo "  test       测试 API 接口"
    echo ""
}

# 检查虚拟环境
check_venv() {
    if [ ! -f "$VENV_PYTHON" ]; then
        echo -e "${RED}错误: 虚拟环境未找到: $VENV_PYTHON${NC}"
        exit 1
    fi
}

# 获取服务 PID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE" 2>/dev/null || echo ""
    else
        # 尝试从进程查找
        pgrep -f "start_server.py" | head -1 || echo ""
    fi
}

# 检查服务是否运行
is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 获取当前日志文件路径
get_log_file() {
    if [ -f "$CURRENT_LOG_FILE" ]; then
        cat "$CURRENT_LOG_FILE" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# 加载环境变量
load_environment() {
    echo -e "${BLUE}加载环境变量...${NC}"

    # 自动检测 Python 版本并设置虚拟环境路径
    PYTHON_VERSION=$(get_python_version)
    VENV_SITE_PACKAGES="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"
    if [ -d "$VENV_SITE_PACKAGES" ]; then
        export PYTHONPATH="$VENV_SITE_PACKAGES:$PYTHONPATH"
        echo -e "${GREEN}✓ 虚拟环境路径设置成功 (Python ${PYTHON_VERSION})${NC}"
    else
        echo -e "${YELLOW}⚠ 虚拟环境路径不存在: $VENV_SITE_PACKAGES${NC}"
    fi
    
    # 设置 CUDA 环境变量（必须在启动 Python 之前设置）
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES${NC}"
    
    # 打印环境信息
    echo -e "${BLUE}环境信息:${NC}"
    echo "Python: $VENV_PYTHON"
    echo "Project root: $PROJECT_DIR"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
}

# 启动服务
start_service() {
    check_venv
    
    if is_running; then
        echo -e "${YELLOW}服务已在运行中 (PID: $(get_pid))${NC}"
        echo -e "访问: http://localhost:$PORT"
        return 0
    fi
    
    echo -e "${BLUE}正在启动 ASD 后端服务...${NC}"
    
    cd "$PROJECT_DIR"
    
    # 加载环境变量
    load_environment
    
    # 生成带时间戳的日志文件名
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local LOG_FILE="$PROJECT_DIR/logs/backend_${timestamp}.log"
    
    # 创建 logs 目录
    mkdir -p "$PROJECT_DIR/logs"
    
    # 保存当前日志文件路径
    echo "$LOG_FILE" > "$CURRENT_LOG_FILE"
    
    # 启动服务（传递 CUDA_VISIBLE_DEVICES 等环境变量）
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    HOST="$HOST" \
    PORT="$PORT" \
    nohup "$VENV_PYTHON" "$PROJECT_DIR/start_server.py" > "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # 保存 PID
    echo $pid > "$PID_FILE"
    
    # 等待服务启动
    echo -n "等待服务启动"
    for i in {1..15}; do
        sleep 1
        echo -n "."
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo ""
            echo -e "${GREEN}✅ 服务启动成功!${NC}"
            echo ""
            echo -e "${BLUE}访问地址:${NC}"
            echo "  前端页面: http://localhost:$PORT/"
            echo "  API文档:  http://localhost:$PORT/docs"
            echo "  健康检查: http://localhost:$PORT/health"
            echo ""
            echo -e "${BLUE}日志文件:${NC} $LOG_FILE"
            echo -e "${BLUE}进程 PID:${NC} $pid"
            return 0
        fi
    done
    
    echo ""
    echo -e "${YELLOW}⚠ 服务启动可能有问题，请检查日志${NC}"
    echo -e "日志文件: $LOG_FILE"
    return 1
}

# 停止服务
stop_service() {
    local pid=$(get_pid)
    
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}服务未运行${NC}"
        return 0
    fi
    
    echo -e "${BLUE}正在停止服务 (PID: $pid)...${NC}"
    
    # 尝试优雅停止
    if kill "$pid" 2>/dev/null; then
        # 等待进程结束
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✅ 服务已停止${NC}"
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 1
        done
        
        # 强制停止
        echo -e "${YELLOW}强制停止服务...${NC}"
        kill -9 "$pid" 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ 服务已停止${NC}"
}

# 查看状态
show_status() {
    if is_running; then
        local pid=$(get_pid)
        echo -e "${GREEN}服务运行中${NC}"
        echo "PID: $pid"
        echo "访问: http://localhost:$PORT"
        
        local log_file=$(get_log_file)
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            echo "日志: $log_file"
        fi
    else
        echo -e "${YELLOW}服务未运行${NC}"
    fi
}

# 查看日志
show_log() {
    local log_file=$(get_log_file)
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo -e "${BLUE}正在查看日志 (Ctrl+C 退出):${NC} $log_file"
        tail -f "$log_file"
    else
        echo -e "${YELLOW}未找到日志文件${NC}"
        # 尝试查找最新的日志
        local latest_log=$(ls -t "$PROJECT_DIR/logs"/backend_*.log 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo -e "${BLUE}正在查看最新日志:${NC} $latest_log"
            tail -f "$latest_log"
        fi
    fi
}

# 测试 API
test_api() {
    echo -e "${BLUE}测试 API 接口...${NC}"
    
    # 测试健康检查
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 健康检查通过${NC}"
    else
        echo -e "${RED}✗ 健康检查失败${NC}"
        return 1
    fi
    
    # 测试设备列表
    local devices=$(curl -s "http://localhost:$PORT/api/detection/devices" 2>/dev/null)
    if [ -n "$devices" ]; then
        echo -e "${GREEN}✓ 设备列表接口正常${NC}"
        echo "设备: $(echo "$devices" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('devices', [])))" 2>/dev/null || echo 'N/A') 个"
    else
        echo -e "${RED}✗ 设备列表接口失败${NC}"
    fi
    
    # 测试参考音频列表
    local refs=$(curl -s "http://localhost:$PORT/api/detection/reference-audios" 2>/dev/null)
    if [ -n "$refs" ]; then
        echo -e "${GREEN}✓ 参考音频接口正常${NC}"
        echo "参考音频: $(echo "$refs" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total', 'N/A'))" 2>/dev/null || echo 'N/A') 个"
    else
        echo -e "${RED}✗ 参考音频接口失败${NC}"
    fi
}

# 主入口
case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 2
        start_service
        ;;
    status)
        show_status
        ;;
    log)
        show_log
        ;;
    test)
        test_api
        ;;
    *)
        show_help
        ;;
esac
