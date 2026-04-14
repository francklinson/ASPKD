#!/bin/bash
# ASD 客户端监控服务管理脚本
# 支持: start | stop | restart | status

set -e

# 配置
CLIENT_DIR="/home/zhouchenghao/PycharmProjects/ASD_for_SPK/client"
PROJECT_DIR="/home/zhouchenghao/PycharmProjects/ASD_for_SPK"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
PID_FILE="$CLIENT_DIR/.client.pid"
LOG_FILE="$CLIENT_DIR/client.log"
SERVER_URL="${ASD_SERVER_URL:-http://localhost:8004}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助
show_help() {
    echo -e "${BLUE}ASD 客户端监控服务管理脚本${NC}"
    echo ""
    echo "用法: ./client.sh [命令]"
    echo ""
    echo "命令:"
    echo "  start      启动客户端"
    echo "  stop       停止客户端"
    echo "  restart    重启客户端"
    echo "  status     查看客户端状态"
    echo "  log        查看实时日志"
    echo "  test       测试与服务端的连接"
    echo ""
}

# 检查虚拟环境
check_venv() {
    if [ ! -f "$VENV_PYTHON" ]; then
        echo -e "${RED}错误: 虚拟环境未找到: $VENV_PYTHON${NC}"
        exit 1
    fi
}

# 获取客户端 PID
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE" 2>/dev/null || echo ""
    else
        # 尝试从进程查找
        pgrep -f "client_monitor.py" | head -1 || echo ""
    fi
}

# 检查客户端是否运行
is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 加载环境变量
load_env() {
    if [ -f "$CLIENT_DIR/.env" ]; then
        echo -e "${BLUE}加载客户端配置...${NC}"
        # 读取监控目录配置
        local monitor_dir=$(grep "ASD_MONITOR_DIR" "$CLIENT_DIR/.env" | cut -d'=' -f2-)
        echo -e "${BLUE}监控目录:${NC} $monitor_dir"
    fi
}

# 启动客户端
start_client() {
    check_venv
    
    if is_running; then
        echo -e "${YELLOW}客户端已在运行中 (PID: $(get_pid))${NC}"
        return 0
    fi
    
    echo -e "${BLUE}正在启动 ASD 客户端监控...${NC}"
    
    cd "$CLIENT_DIR"
    
    # 加载环境变量
    load_env
    
    # 清空旧日志
    > "$LOG_FILE"
    
    # 启动客户端
    nohup "$VENV_PYTHON" "$CLIENT_DIR/client_monitor.py" > "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # 保存 PID
    echo $pid > "$PID_FILE"
    
    # 等待客户端启动
    echo -n "等待客户端启动"
    for i in {1..15}; do
        sleep 1
        echo -n "."
        
        # 检查进程是否存在
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo -e "${RED}❌ 客户端启动失败${NC}"
            echo -e "查看日志: tail -50 $LOG_FILE"
            rm -f "$PID_FILE"
            return 1
        fi
        
        # 检查是否注册成功
        if grep -q "客户端注册成功" "$LOG_FILE" 2>/dev/null; then
            echo ""
            echo -e "${GREEN}✅ 客户端启动成功!${NC}"
            echo ""
            echo -e "${BLUE}客户端信息:${NC}"
            
            # 提取客户端ID
            local client_id=$(grep "客户端注册成功" "$LOG_FILE" | tail -1 | grep -oE '[a-f0-9]{12}' | head -1)
            if [ -n "$client_id" ]; then
                echo "  客户端ID: $client_id"
            fi
            
            echo "  服务端: $SERVER_URL"
            echo "  日志文件: $LOG_FILE"
            echo -e "  进程 PID: ${GREEN}$pid${NC}"
            echo ""
            echo -e "${BLUE}查看日志:${NC} tail -f $LOG_FILE"
            return 0
        fi
        
        # 检查是否出错
        if grep -q "注册失败\|ERROR" "$LOG_FILE" 2>/dev/null; then
            echo ""
            echo -e "${RED}❌ 客户端注册失败${NC}"
            echo -e "查看日志: tail -30 $LOG_FILE"
            rm -f "$PID_FILE"
            return 1
        fi
    done
    
    echo ""
    echo -e "${YELLOW}⚠ 客户端启动超时，请检查日志${NC}"
    echo -e "查看日志: tail -50 $LOG_FILE"
    return 1
}

# 停止客户端
stop_client() {
    local pid=$(get_pid)
    
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}客户端未运行${NC}"
        # 清理可能残留的进程
        pkill -f "client_monitor.py" 2>/dev/null || true
        rm -f "$PID_FILE"
        return 0
    fi
    
    echo -e "${BLUE}正在停止客户端 (PID: $pid)...${NC}"
    
    # 尝试优雅停止
    if kill -0 "$pid" 2>/dev/null; then
        # 先发送断开通知（如果有）
        kill "$pid" 2>/dev/null || true
        
        # 等待进程结束
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo -e "${GREEN}✅ 客户端已停止${NC}"
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 0.5
        done
        
        # 强制停止
        echo -e "${YELLOW}强制停止...${NC}"
        kill -9 "$pid" 2>/dev/null || true
        pkill -9 -f "client_monitor.py" 2>/dev/null || true
    fi
    
    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ 客户端已停止${NC}"
}

# 重启客户端
restart_client() {
    echo -e "${BLUE}重启客户端...${NC}"
    stop_client
    sleep 2
    start_client
}

# 查看状态
show_status() {
    local pid=$(get_pid)
    
    echo -e "${BLUE}=== 客户端状态 ===${NC}"
    echo ""
    
    if is_running; then
        echo -e "状态: ${GREEN}运行中${NC}"
        echo "PID: $pid"
        
        # 获取进程信息
        ps -p "$pid" -o pid,ppid,cmd,etime 2>/dev/null || true
        
        # 从日志提取信息
        if [ -f "$LOG_FILE" ]; then
            # 客户端ID
            local client_id=$(grep "客户端注册成功" "$LOG_FILE" | tail -1 | grep -oE '[a-f0-9]{12}' | head -1)
            if [ -n "$client_id" ]; then
                echo ""
                echo "客户端ID: $client_id"
            fi
            
            # 监控目录
            local monitor_dir=$(grep "正在监控目录" "$LOG_FILE" | tail -1 | grep -oP '正在监控目录: \K.*')
            if [ -n "$monitor_dir" ]; then
                echo "监控目录: $monitor_dir"
            fi
            
            # 上传统计
            local uploaded=$(grep -c "上传成功" "$LOG_FILE" 2>/dev/null || echo "0")
            if [ "$uploaded" != "0" ] && [ "$uploaded" -gt 0 ]; then
                echo "上传成功: $uploaded 个文件"
            fi
        fi
        
        # 测试服务端连接
        echo ""
        echo -n "服务端连接: "
        if curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
            echo -e "${GREEN}正常${NC}"
        else
            echo -e "${RED}无法连接${NC}"
        fi
        
    else
        echo -e "状态: ${RED}未运行${NC}"
        
        # 检查是否有残留进程
        local residual=$(pgrep -f "client_monitor.py" | head -1)
        if [ -n "$residual" ]; then
            echo -e "${YELLOW}发现残留进程: $residual${NC}"
            echo "建议运行: ./client.sh stop"
        fi
    fi
    
    # 日志文件大小
    if [ -f "$LOG_FILE" ]; then
        echo ""
        local log_size=$(du -h "$LOG_FILE" 2>/dev/null | cut -f1)
        echo -e "日志文件: $LOG_FILE ($log_size)"
    fi
}

# 查看日志
show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}日志文件不存在${NC}"
        return 1
    fi
    
    echo -e "${BLUE}正在监听日志: $LOG_FILE (按 Ctrl+C 退出)...${NC}"
    echo ""
    tail -f "$LOG_FILE"
}

# 测试连接
test_connection() {
    echo -e "${BLUE}=== 连接测试 ===${NC}"
    echo ""
    
    # 测试服务端健康检查
    echo -n "服务端健康检查: "
    local health=$(curl -s "$SERVER_URL/health" 2>/dev/null)
    if [ -n "$health" ]; then
        echo -e "${GREEN}✓${NC} $health"
    else
        echo -e "${RED}✗ 失败${NC}"
    fi
    
    # 测试客户端注册接口
    echo -n "客户端注册接口: "
    local register=$(curl -s -X POST "$SERVER_URL/api/client/register" -H "Content-Type: application/json" -d '{"client_name":"test"}' 2>/dev/null)
    if [ -n "$register" ] && echo "$register" | grep -q "success"; then
        echo -e "${GREEN}✓${NC} 正常"
    else
        echo -e "${RED}✗ 失败${NC}"
    fi
    
    # 测试客户端状态接口
    echo -n "客户端状态接口: "
    local status=$(curl -s "$SERVER_URL/api/client/status" 2>/dev/null)
    if [ -n "$status" ]; then
        local count=$(echo "$status" | grep -o '"client_id"' | wc -l)
        echo -e "${GREEN}✓${NC} $count 个客户端在线"
    else
        echo -e "${RED}✗ 失败${NC}"
    fi
}

# 主逻辑
case "${1:-}" in
    start)
        start_client
        ;;
    stop)
        stop_client
        ;;
    restart|reboot)
        restart_client
        ;;
    status)
        show_status
        ;;
    log|logs)
        show_logs
        ;;
    test)
        test_connection
        ;;
    *)
        show_help
        exit 1
        ;;
esac
