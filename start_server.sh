#!/bin/bash
# ASD 音频异常检测服务管理脚本
# 支持: start | stop | restart | status

set -e

# 配置
PROJECT_DIR="/home/zhouchenghao/PycharmProjects/ASD_for_SPK"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PID_FILE="$PROJECT_DIR/.service.pid"
CURRENT_LOG_FILE="$PROJECT_DIR/.current_log"  # 记录当前日志文件路径
HOST="0.0.0.0"
PORT="8004"

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
    
    # 设置虚拟环境路径
    VENV_SITE_PACKAGES="$PROJECT_DIR/.venv/lib/python3.12/site-packages"
    if [ -d "$VENV_SITE_PACKAGES" ]; then
        export PYTHONPATH="$VENV_SITE_PACKAGES:$PYTHONPATH"
        echo -e "${GREEN}✓ 虚拟环境路径设置成功${NC}"
    else
        echo -e "${YELLOW}⚠ 虚拟环境路径不存在: $VENV_SITE_PACKAGES${NC}"
    fi
    
    # 加载配置文件中的环境变量
    CONFIG_PATH="$PROJECT_DIR/config/config.yaml"
    if [ -f "$CONFIG_PATH" ]; then
        echo -e "${BLUE}加载配置文件: $CONFIG_PATH${NC}"
        # 使用 Python 解析 YAML 并设置环境变量
        "$VENV_PYTHON" - "$CONFIG_PATH" << 'EOF'
import yaml
import os
import sys

config_path = sys.argv[1]
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    env_config = config.get('environments', {})
    for key, value in env_config.items():
        if value and key not in os.environ:
            os.environ[key] = str(value)
            print(f"[Config] Set env: {key}={value}")
except Exception as e:
    print(f"[Config] Warning: Failed to load environment variables: {e}")
EOF
    else
        echo -e "${YELLOW}⚠ 配置文件不存在: $CONFIG_PATH${NC}"
    fi
    
    # 打印环境信息
    echo -e "${BLUE}环境信息:${NC}"
    echo "Python: $VENV_PYTHON"
    echo "Project root: $PROJECT_DIR"
    echo "Site-packages: $VENV_SITE_PACKAGES"
    echo "Python path: $PYTHONPATH"
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
    
    # 启动服务
    HOST="$HOST" PORT="$PORT" nohup "$VENV_PYTHON" "$PROJECT_DIR/start_server.py" > "$LOG_FILE" 2>&1 &
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
    echo -e "${RED}❌ 服务启动失败或超时${NC}"
    echo -e "查看日志: tail -50 $LOG_FILE"
    return 1
}

# 停止服务
stop_service() {
    local pid=$(get_pid)
    
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}服务未运行${NC}"
        # 清理可能残留的进程
        pkill -f "start_server.py" 2>/dev/null || true
        rm -f "$PID_FILE"
        return 0
    fi
    
    echo -e "${BLUE}正在停止服务 (PID: $pid)...${NC}"
    
    # 尝试优雅停止
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        
        # 等待进程结束
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✅ 服务已停止${NC}"
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 0.5
        done
        
        # 强制停止
        echo -e "${YELLOW}强制停止...${NC}"
        kill -9 "$pid" 2>/dev/null || true
        pkill -9 -f "start_server.py" 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ 服务已停止${NC}"
}

# 重启服务
restart_service() {
    echo -e "${BLUE}重启服务...${NC}"
    stop_service
    sleep 2
    start_service
}

# 查看状态
show_status() {
    local pid=$(get_pid)
    local LOG_FILE=$(get_log_file)
    
    echo -e "${BLUE}=== 服务状态 ===${NC}"
    echo ""
    
    if is_running; then
        echo -e "状态: ${GREEN}运行中${NC}"
        echo "PID: $pid"
        
        # 获取进程信息
        ps -p "$pid" -o pid,ppid,cmd,etime 2>/dev/null || true
        
        # 测试 API
        echo ""
        echo -n "API 测试: "
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            local health=$(curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
            echo -e "${GREEN}正常 ($health)${NC}"
        else
            echo -e "${RED}无法连接${NC}"
        fi
        
        echo ""
        echo -e "${BLUE}访问地址:${NC}"
        echo "  前端: http://localhost:$PORT/"
        echo "  API:  http://localhost:$PORT/docs"
        
        # 端口占用情况
        echo ""
        echo -e "${BLUE}端口 $PORT 占用:${NC}"
        netstat -tlnp 2>/dev/null | grep ":$PORT" || ss -tlnp 2>/dev/null | grep ":$PORT" || echo "  (无法获取)"
        
    else
        echo -e "状态: ${RED}未运行${NC}"
        
        # 检查是否有残留进程
        local residual=$(pgrep -f "start_server.py" | head -1)
        if [ -n "$residual" ]; then
            echo -e "${YELLOW}发现残留进程: $residual${NC}"
            echo "建议运行: ./service.sh stop"
        fi
    fi
    
    # 日志文件大小
    if [ -f "$LOG_FILE" ]; then
        echo ""
        local log_size=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1)
        echo -e "${BLUE}当前日志文件:${NC} $LOG_FILE ($log_size)"
    fi
    
    # 显示日志目录统计
    if [ -d "$PROJECT_DIR/logs" ]; then
        echo ""
        local log_count=$(ls -1 "$PROJECT_DIR/logs"/backend_*.log 2>/dev/null | wc -l)
        echo -e "${BLUE}历史日志文件:${NC} $log_count 个"
        echo "  目录: $PROJECT_DIR/logs/"
    fi
}

# 查看日志
show_logs() {
    local LOG_FILE=$(get_log_file)
    
    if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
        # 尝试查找最新的日志文件
        LOG_FILE=$(ls -t "$PROJECT_DIR/logs"/backend_*.log 2>/dev/null | head -1)
        if [ -z "$LOG_FILE" ]; then
            echo -e "${YELLOW}日志文件不存在${NC}"
            return 1
        fi
    fi
    
    echo -e "${BLUE}正在监听日志: $LOG_FILE (按 Ctrl+C 退出)...${NC}"
    echo ""
    tail -f "$LOG_FILE"
}

# 测试 API
test_api() {
    echo -e "${BLUE}=== API 接口测试 ===${NC}"
    echo ""
    
    local base_url="http://localhost:$PORT"
    
    # 测试健康检查
    echo -n "健康检查: "
    local health=$(curl -s "$base_url/health" 2>/dev/null)
    if [ -n "$health" ]; then
        echo -e "${GREEN}✓${NC} $health"
    else
        echo -e "${RED}✗ 失败${NC}"
    fi
    
    # 测试算法列表
    echo -n "算法列表: "
    local algos=$(curl -s "$base_url/api/detection/algorithms" 2>/dev/null)
    if [ -n "$algos" ]; then
        local count=$(echo "$algos" | grep -o '"name"' | wc -l)
        echo -e "${GREEN}✓${NC} 已注册 $count 个算法"
    else
        echo -e "${RED}✗ 失败${NC}"
    fi
    
    # 测试任务列表
    echo -n "任务列表: "
    local tasks=$(curl -s "$base_url/api/tasks/list" 2>/dev/null)
    if [ -n "$tasks" ]; then
        local count=$(echo "$tasks" | grep -o '"task_id"' | wc -l)
        echo -e "${GREEN}✓${NC} $count 个任务"
    else
        echo -e "${RED}✗ 失败${NC}"
    fi
    
    # 测试监控状态
    echo -n "监控状态: "
    local monitor=$(curl -s "$base_url/api/monitor/status" 2>/dev/null)
    if [ -n "$monitor" ]; then
        local status=$(echo "$monitor" | grep -o '"is_monitoring":[^,}]*' | cut -d':' -f2)
        if [ "$status" = "true" ]; then
            echo -e "${GREEN}✓${NC} 运行中"
        else
            echo -e "${YELLOW}○${NC} 未启动"
        fi
    else
        echo -e "${RED}✗ 失败${NC}"
    fi
}

# 主逻辑
case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart|reboot)
        restart_service
        ;;
    status)
        show_status
        ;;
    log|logs)
        show_logs
        ;;
    test)
        test_api
        ;;
    *)
        show_help
        exit 1
        ;;
esac
