#!/bin/bash
# ASD 音频异常检测服务管理脚本
# 支持: start | stop | restart | status

# 不要在函数内部退出，允许函数返回错误

# 配置
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_step() {
    local step=$1
    local total=$2
    local msg=$3
    echo -e "${BLUE}[$step/$total]${NC} $msg"
}

# 检查目录
check_directories() {
    # 确保 PROJECT_DIR 已设置
    if [ -z "$PROJECT_DIR" ]; then
        PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
    
    print_step 1 6 "检查项目环境"
    print_info "项目目录: $PROJECT_DIR"
    
    local dirs=("backend" "algorithms" "data" "models" "uploads" "output" "logs")
    for dir in "${dirs[@]}"; do
        if [ -d "$PROJECT_DIR/$dir" ]; then
            print_success "目录存在: $dir/"
        else
            print_warning "创建目录: $dir/"
            mkdir -p "$PROJECT_DIR/$dir"
        fi
    done
    
    # 检查配置文件
    if [ -f "$PROJECT_DIR/backend/config/config.yaml" ]; then
        print_success "配置文件: backend/config/config.yaml"
    else
        print_error "配置文件不存在: backend/config/config.yaml"
        exit 1
    fi
}

# 检查GPU
check_gpu() {
    print_step 2 6 "检查GPU状态"
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null)
        if [ -n "$gpu_info" ]; then
            local gpu_count=$(echo "$gpu_info" | wc -l)
            print_success "检测到 $gpu_count 个GPU"
            
            local i=0
            while IFS=',' read -r name mem_total mem_free; do
                name=$(echo "$name" | xargs)
                mem_total=$(echo "$mem_total" | xargs)
                mem_free=$(echo "$mem_free" | xargs)
                echo "  GPU $i: $name"
                echo "        显存: $mem_free / $mem_total"
                ((i++))
            done <<< "$gpu_info"
        else
            print_warning "nvidia-smi 执行失败"
        fi
    else
        print_warning "nvidia-smi 未找到，将使用CPU模式"
    fi
}

# 检查Python依赖
check_dependencies() {
    print_step 3 6 "检查Python依赖"
    
    # 使用关联数组存储模块名和导入名（可能不同）
    declare -A modules
    modules["librosa"]="librosa"
    modules["fastapi"]="fastapi"
    modules["uvicorn"]="uvicorn"
    modules["torch"]="torch"
    modules["numpy"]="numpy"
    modules["soundfile"]="soundfile"
    modules["pydantic"]="pydantic"
    modules["yaml"]="yaml"
    
    local failed=()
    
    for name in "${!modules[@]}"; do
        local import_name="${modules[$name]}"
        if $VENV_PYTHON -c "import $import_name" 2>/dev/null; then
            local version=$($VENV_PYTHON -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null)
            print_success "$name ($version)"
        else
            print_error "$name 未安装"
            failed+=("$name")
        fi
    done
    
    if [ ${#failed[@]} -gt 0 ]; then
        print_error "缺少依赖: ${failed[*]}"
        exit 1
    fi
}

# 检查数据库
check_database() {
    print_step 4 6 "检查数据库连接"
    
    # 尝试导入并测试连接
    local db_output=$($VENV_PYTHON -c "
import sys
import io
sys.path.insert(0, '$PROJECT_DIR')

# 捕获所有输出
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()

try:
    from backend.core.shazam.database.connector import DatabaseChecker, MySQLConnector
    from backend.core.shazam.utils.hparam import hp
    
    # 先检查并创建数据库和表
    checker = DatabaseChecker()
    checker.check_database()
    checker.check_tables()
    
    # 然后测试连接
    conn = MySQLConnector()
    conn.cursor.execute('SELECT COUNT(*) FROM finger_prints')
    count = conn.cursor.fetchone()[0]
    conn.cursor.close()
    conn.conn.close()
    
    # 恢复输出并打印结果
    sys.stdout = old_stdout
    print(f'OK|{count}')
except Exception as e:
    sys.stdout = old_stdout
    print(f'ERROR|{e}')
" 2>&1)
    
    # 提取最后一行作为结果
    local db_test=$(echo "$db_output" | tail -1)
    
    if [[ "$db_test" == OK* ]]; then
        local count=$(echo "$db_test" | cut -d'|' -f2)
        print_success "MySQL连接成功"
        echo "  数据库: music_recognition"
        echo "  指纹库记录数: $count"
    else
        local error=$(echo "$db_test" | cut -d'|' -f2)
        print_warning "数据库检查: $error"
        print_info "将在首次使用时自动创建数据库和表"
    fi
}

# 检查模型文件
check_models() {
    print_step 5 6 "检查模型文件"
    
    local model_dirs=("$PROJECT_DIR/models/pre_trained" "$PROJECT_DIR/models/saved")
    
    for model_dir in "${model_dirs[@]}"; do
        local dir_name=$(basename "$model_dir")
        if [ -d "$model_dir" ]; then
            local pth_count=$(find "$model_dir" -name "*.pth" 2>/dev/null | wc -l)
            if [ $pth_count -gt 0 ]; then
                print_success "$dir_name: $pth_count 个模型"
                # 显示前3个
                find "$model_dir" -name "*.pth" -exec basename {} \; 2>/dev/null | head -3 | sed 's/^/  - /'
                if [ $pth_count -gt 3 ]; then
                    echo "  ... 还有 $((pth_count - 3)) 个模型"
                fi
            else
                print_warning "$dir_name: 暂无模型"
            fi
        else
            print_warning "创建目录: $dir_name"
            mkdir -p "$model_dir"
        fi
    done
    
    # 检查关键基础模型（backbone）
    echo ""
    print_info "检查关键基础模型..."
    
    local required_models=(
        "dinov2_vits14_pretrain.pth:基础backbone"
        "dinov2_vitb14_pretrain.pth:基础backbone"
        "dinov2_vitl14_pretrain.pth:基础backbone"
        "wide_resnet50_2-95faca4d.pth:基础backbone"
        "resnet18-f37072fd.pth:基础backbone"
    )
    
    local missing_required=()
    for item in "${required_models[@]}"; do
        local model_file=$(echo "$item" | cut -d':' -f1)
        local model_desc=$(echo "$item" | cut -d':' -f2)
        
        if [ -f "$PROJECT_DIR/models/pre_trained/$model_file" ]; then
            print_success "$model_desc: $model_file"
        else
            print_warning "$model_desc: $model_file 缺失"
            missing_required+=("$model_file")
        fi
    done
    
    # 检查算法特定模型（仅警告，不阻止启动）
    echo ""
    print_info "检查算法训练模型（可选）..."
    
    local algorithm_models=(
        "dinomaly_dinov2_small.pth:Dinomaly算法"
        "dinomaly_dinov3_small.pth:Dinomaly算法"
        "mambaad_best.pth:MambaAD算法"
        "patchcore_best.pth:PatchCore算法"
    )
    
    for item in "${algorithm_models[@]}"; do
        local model_file=$(echo "$item" | cut -d':' -f1)
        local model_desc=$(echo "$item" | cut -d':' -f2)
        
        if [ -f "$PROJECT_DIR/models/pre_trained/$model_file" ]; then
            print_success "$model_desc: $model_file"
        else
            print_info "$model_desc: $model_file 未找到（将在首次训练时生成）"
        fi
    done
    
    if [ ${#missing_required[@]} -gt 0 ]; then
        echo ""
        print_warning "缺少关键基础模型，部分功能可能无法正常使用"
        print_info "请从以下地址下载预训练模型:"
        print_info "  - DINOv2: https://github.com/facebookresearch/dinov2"
        print_info "  - ResNet: https://download.pytorch.org/models/"
    fi
}

# 检查端口
check_port() {
    print_step 6 6 "检查端口"
    
    if command -v lsof &> /dev/null; then
        if lsof -i :$PORT &> /dev/null; then
            print_error "端口 $PORT 已被占用"
            echo "  占用进程:"
            lsof -i :$PORT | grep -v COMMAND | head -3
            exit 1
        else
            print_success "端口 $PORT 可用"
        fi
    else
        print_warning "无法检查端口 (lsof 未安装)"
    fi
}

# 加载环境变量
load_environment() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  加载环境配置${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # 自动检测 Python 版本并设置虚拟环境路径
    PYTHON_VERSION=$(get_python_version)
    VENV_SITE_PACKAGES="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages"
    if [ -d "$VENV_SITE_PACKAGES" ]; then
        export PYTHONPATH="$VENV_SITE_PACKAGES:$PYTHONPATH"
        print_success "虚拟环境路径 (Python ${PYTHON_VERSION})"
    else
        print_warning "虚拟环境路径不存在: $VENV_SITE_PACKAGES"
    fi
    
    # 设置 CUDA 环境变量（必须在启动 Python 之前设置）
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    print_info "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # 打印环境信息
    echo ""
    echo "Python: $VENV_PYTHON"
    echo "Project: $PROJECT_DIR"
    echo "Port: $PORT"
}

# 启动服务
start_service() {
    check_venv
    
    if is_running; then
        echo -e "${YELLOW}服务已在运行中 (PID: $(get_pid))${NC}"
        echo -e "访问: http://localhost:$PORT"
        return 0
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  🎵 ASD 音频异常检测系统启动${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    cd "$PROJECT_DIR"
    
    # 执行启动前检查
    check_directories
    check_gpu
    check_dependencies
    check_database
    check_models
    check_port
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ 所有检查通过，准备启动服务${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
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
    echo ""
    echo -n "等待服务启动"
    local start_time=$(date +%s)
    for i in {1..30}; do
        sleep 1
        echo -n "."
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            local end_time=$(date +%s)
            local elapsed=$((end_time - start_time))
            echo ""
            echo ""
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${GREEN}  ✅ 服务启动成功! (耗时 ${elapsed}秒)${NC}"
            echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo ""
            echo -e "${BLUE}访问地址:${NC}"
            echo "  🌐 前端页面: http://localhost:$PORT/"
            echo "  📚 API文档:  http://localhost:$PORT/docs"
            echo "  💓 健康检查: http://localhost:$PORT/health"
            echo ""
            echo -e "${BLUE}管理命令:${NC}"
            echo "  查看日志: ./start_server.sh log"
            echo "  查看状态: ./start_server.sh status"
            echo "  停止服务: ./start_server.sh stop"
            echo ""
            echo -e "${BLUE}日志文件:${NC} $LOG_FILE"
            echo -e "${BLUE}进程 PID:${NC} $pid"
            echo ""
            return 0
        fi
    done
    
    echo ""
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ⚠ 服务启动超时，请检查日志${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "日志文件: $LOG_FILE"
    echo -e "查看日志: tail -f $LOG_FILE"
    return 1
}

# 检查GPU显存
check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)
        if [ -n "$gpu_info" ]; then
            echo ""
            print_info "GPU显存状态:"
            local i=0
            while IFS=',' read -r name mem_used mem_total; do
                name=$(echo "$name" | xargs)
                mem_used=$(echo "$mem_used" | xargs)
                mem_total=$(echo "$mem_total" | xargs)
                echo "  GPU $i: $name"
                echo "        显存使用: $mem_used / $mem_total"
                ((i++))
            done <<< "$gpu_info"
        fi
    fi
}

# 停止服务
stop_service() {
    local pid=$(get_pid)
    
    if [ -z "$pid" ]; then
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}  ℹ 服务未运行${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        return 0
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  🛑 停止 ASD 音频异常检测系统${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "停止时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 获取进程信息
    local process_info=$(ps -p "$pid" -o comm=,etime= 2>/dev/null || echo "unknown")
    print_info "目标进程: $process_info"
    print_info "进程 PID: $pid"
    
    # 停止前检查显存
    check_gpu_memory
    echo ""
    
    print_step 1 4 "发送终止信号"
    
    # 尝试优雅停止
    if kill "$pid" 2>/dev/null; then
        print_success "已发送 SIGTERM 信号"
        
        print_step 2 4 "等待进程结束"
        local start_wait=$(date +%s)
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                local end_wait=$(date +%s)
                local wait_time=$((end_wait - start_wait))
                print_success "进程已结束 (耗时 ${wait_time}秒)"
                
                print_step 3 4 "清理资源"
                rm -f "$PID_FILE"
                print_success "已移除 PID 文件"
                
                print_step 4 4 "检查显存释放"
                sleep 1  # 等待显存释放
                check_gpu_memory
                
                echo ""
                echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo -e "${GREEN}  ✅ 服务已成功停止${NC}"
                echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
                echo ""
                return 0
            fi
            echo -n "."
            sleep 1
        done
        echo ""
        
        # 强制停止
        print_warning "优雅停止超时，执行强制停止"
        print_step 3 4 "强制终止进程"
        if kill -9 "$pid" 2>/dev/null; then
            print_success "已发送 SIGKILL 信号"
        else
            print_warning "进程可能已经退出"
        fi
    else
        print_error "无法终止进程 (可能已无权限)"
    fi
    
    print_step 4 4 "检查显存释放"
    sleep 1
    check_gpu_memory
    
    rm -f "$PID_FILE"
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✅ 服务已停止${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
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
