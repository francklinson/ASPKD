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
    
    local dirs=("backend" "algorithms" "data" "models" "data/uploads" "data/output" "data/output/vis" "data/spk" "data/dataset_builder" "data/ref" "logs")
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
    print_step 4 6 "初始化Shazam内存数据库"

    # 初始化内存数据库（无需外部数据库服务）
    local db_output=$($VENV_PYTHON -c "
import sys
import io
sys.path.insert(0, '$PROJECT_DIR')

old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()

try:
    from backend.core.shazam.database.in_memory import InMemoryDatabaseChecker, _MemDB

    checker = InMemoryDatabaseChecker()
    checker.check_database()
    checker.check_tables()

    stats = _MemDB().stats()

    sys.stdout = old_stdout
    print(f'OK|{stats[\"music_count\"]}|{stats[\"total_hashes\"]}|{stats[\"unique_hashes\"]}')
except Exception as e:
    sys.stdout = old_stdout
    print(f'ERROR|{e}')
" 2>&1)

    local db_test=$(echo "$db_output" | tail -1)

    if [[ "$db_test" == OK* ]]; then
        local music_count=$(echo "$db_test" | cut -d'|' -f2)
        local total_hashes=$(echo "$db_test" | cut -d'|' -f3)
        local unique_hashes=$(echo "$db_test" | cut -d'|' -f4)
        print_success "Shazam 内存数据库已就绪（无外部依赖）"
        echo "  已加载曲目: $music_count 首"
        echo "  总指纹数: $total_hashes"
        echo "  唯一哈希: $unique_hashes"
    else
        local error=$(echo "$db_test" | cut -d'|' -f2)
        print_error "内存数据库初始化失败: $error"
        print_info "请检查 Shazam 模块配置"
        exit 1
    fi
}

# 检查模型文件
check_models() {
    print_step 5 6 "检查模型文件"

    local PRETRAINED_DIR="$PROJECT_DIR/models/pre_trained"
    local SAVED_DIR="$PROJECT_DIR/models/saved"

    # ── 辅助函数：计算相对路径 ──
    _rel_path() {
        local full="$1"
        echo "${full#$PROJECT_DIR/}"
    }

    # ── 辅助函数：打印单行模型信息（路径/大小/类型） ──
    _model_line() {
        # $1: 文件完整路径  $2: 描述标签
        local f="$1" tag="$2"
        local rp=$(_rel_path "$f")
        if [ ! -e "$f" ]; then
            echo -e "  ${YELLOW}⚠${NC} ${tag}  ${YELLOW}[缺失]  ${rp}${NC}"
            return 1
        fi
        local sz=$(du -h "$f" 2>/dev/null | cut -f1)
        local icon="📄"
        local link=""
        if [ -L "$f" ]; then
            icon="🔗"
            local target
            target=$(readlink "$f")
            link=" → ${target##*/}"
        fi
        echo -e "  ${GREEN}✓${NC} ${icon} ${BLUE}${rp}${NC}  ${sz}${link}"
        return 0
    }

    _ls_model() {
        # 列出单个文件（缩进版）
        local f="$1"
        local rp=$(_rel_path "$f")
        local sz=$(du -h "$f" 2>/dev/null | cut -f1)
        local icon=" "
        local link=""
        if [ -L "$f" ]; then
            icon="🔗"
            link=" → $(basename "$(readlink "$f")")"
        fi
        echo -e "    ${icon} ${rp}  ${sz}${link}"
    }

    # ═══════════════════════════════════════════
    # 1. 目录概览 + 列出所有文件
    # ═══════════════════════════════════════════
    for model_dir in "$PRETRAINED_DIR" "$SAVED_DIR"; do
        local dir_name=$(basename "$model_dir")
        if [ -d "$model_dir" ]; then
            local files=()
            while IFS= read -r -d '' f; do files+=("$f"); done < \
                <(find "$model_dir" -maxdepth 1 \( -name "*.pth" -o -name "*.pt" -o -name "*.pckl" \) -print0 2>/dev/null | sort -z)
            local total=${#files[@]}
            local total_sz=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
            if [ $total -gt 0 ]; then
                echo -e "  ${GREEN}✓${NC} ${BLUE}${dir_name}/${NC} — ${total} 个权重文件, ${total_sz}"
                for f in "${files[@]}"; do _ls_model "$f"; done
            else
                echo -e "  ${YELLOW}⚠${NC} ${dir_name}/: 暂无文件"
            fi
        else
            echo -e "  ${YELLOW}⚠${NC} 创建目录: ${dir_name}/"
            mkdir -p "$model_dir"
        fi
    done

    # efficientad 子目录
    local ef_dir="$PRETRAINED_DIR/efficientad_pretrained_weights"
    if [ -d "$ef_dir" ]; then
        local ef_files=()
        while IFS= read -r -d '' f; do ef_files+=("$f"); done < \
            <(find "$ef_dir" -maxdepth 1 -name "*.pth" -print0 2>/dev/null | sort -z)
        if [ ${#ef_files[@]} -gt 0 ]; then
            local ef_sz=$(du -sh "$ef_dir" 2>/dev/null | cut -f1)
            echo -e "  ${GREEN}✓${NC} ${BLUE}efficientad_pretrained_weights/${NC} — ${#ef_files[@]} 个文件, ${ef_sz}"
            for f in "${ef_files[@]}"; do _ls_model "$f"; done
        fi
    fi

    # ═══════════════════════════════════════════
    # 2. 关键基础模型
    # ═══════════════════════════════════════════
    echo ""
    echo -e "  ${BLUE}▸ 关键基础模型（缺失 → 阻止启动）${NC}"
    echo ""

    local required_models=(
        "dinov2_vits14_pretrain.pth:DINOv2 ViT-Small/14"
        "dinov2_vitb14_pretrain.pth:DINOv2 ViT-Base/14"
        "dinov2_vitl14_pretrain.pth:DINOv2 ViT-Large/14"
        "wide_resnet50_2-95faca4d.pth:Wide ResNet-50-2"
        "resnet18-f37072fd.pth:ResNet-18"
        "dinov2_vits14_reg4_pretrain.pth:DINOv2-S/14+Reg4 (SubspaceAD)"
        "dinov2_vitb14_reg4_pretrain.pth:DINOv2-B/14+Reg4 (SubspaceAD)"
        "dinov2_vitl14_reg4_pretrain.pth:DINOv2-L/14+Reg4 (SubspaceAD)"
    )

    local missing_required=()
    for entry in "${required_models[@]}"; do
        local fname="${entry%%:*}"
        local label="${entry#*:}"
        if _model_line "$PRETRAINED_DIR/$fname" "$label"; then
            :
        else
            missing_required+=("$fname")
        fi
    done

    # ═══════════════════════════════════════════
    # 3. SubspaceAD transformers 模型（本地目录）
    # ═══════════════════════════════════════════
    echo ""
    echo -e "  ${BLUE}▸ SubspaceAD 少样本模型（本地目录）${NC}"
    echo ""

    local DINOV2_DIR="$PRETRAINED_DIR/dinov2"
    if [ -d "$DINOV2_DIR" ]; then
        local dinov2_backbones=(
            "dinov2-with-registers-small:DINOv2-S/14+Reg4 (384dim)"
            "dinov2-with-registers-base:DINOv2-B/14+Reg4  (768dim)"
            "dinov2-with-registers-large:DINOv2-L/14+Reg4 (1024dim)"
        )
        for entry in "${dinov2_backbones[@]}"; do
            local dirname="${entry%%:*}"
            local label="${entry#*:}"
            local dirpath="$DINOV2_DIR/$dirname"
            if [ -d "$dirpath" ]; then
                local has_cfg=0; [ -f "$dirpath/config.json" ] && has_cfg=1
                local has_proc=0; [ -f "$dirpath/preprocessor_config.json" ] && has_proc=1
                local has_weights=0
                if [ -f "$dirpath/model.safetensors" ]; then
                    has_weights=1
                elif [ -f "$dirpath/pytorch_model.bin" ]; then
                    has_weights=1
                fi
                local sz=$(du -sh "$dirpath" 2>/dev/null | cut -f1)
                if [ $has_cfg -eq 1 ] && [ $has_proc -eq 1 ] && [ $has_weights -eq 1 ]; then
                    echo -e "  ${GREEN}✓${NC} ${BLUE}dinov2/$dirname/${NC}  ${sz}  ${label}"
                else
                    echo -e "  ${YELLOW}⚠${NC} ${YELLOW}dinov2/$dirname/${NC}  ${sz}  [不完整]"
                fi
            else
                echo -e "  ${YELLOW}⚠${NC} 缺失 ${YELLOW}dinov2/$dirname/${NC}  — ${label}"
            fi
        done
    else
        echo -e "  ${YELLOW}⚠${NC} dinov2/ 目录不存在，SubspaceAD 少样本检测不可用"
    fi

    # ═══════════════════════════════════════════
    # 4. 算法训练模型（可选）
    # ═══════════════════════════════════════════
    echo ""
    echo -e "  ${BLUE}▸ 算法训练模型（可选，首次使用时下载/训练）${NC}"
    echo ""

    local algorithm_models=(
        "dinomaly_dinov2_small.pth:Dinomaly DINOv2 Small (自训练)"
        "dinomaly_dinov3_small.pth:Dinomaly DINOv3 Small (自训练)"
        "mambaad_best.pth:MambaAD 状态空间模型"
        "invad_best.pth:InVad 生成式模型"
        "vitad_best.pth:ViTAD Transformer"
        "unad_best.pth:UniAD 统一架构"
        "cflow_best.pth:CFlow 归一化流"
        "pyramidflow_best.pth:PyramidFlow 金字塔流"
        "simplenet_best.pth:SimpleNet 特征学习"
        "patchcore_best.pth:PatchCore 特征嵌入 (Anomalib)"
        "efficientad_best.pth:EfficientAD 轻量级 (Anomalib)"
        "padim_best.pth:PaDiM 特征嵌入 (Anomalib)"
        "denseae_best.pth:DenseAE 自编码器 (BaseASD)"
        "cae_best.pth:CAE 自编码器 (BaseASD)"
        "vae_best.pth:VAE 自编码器 (BaseASD)"
        "ViT-B-32.pt:MuSc 零样本 — CLIP ViT-B/32"
        "ViT-B-16.pt:MuSc 零样本 — CLIP ViT-B/16"
        "ViT-L-14.pt:MuSc 零样本 — CLIP ViT-L/14"
        "ViT-L-14-336px.pt:MuSc 零样本 — CLIP ViT-L/14@336px"
    )

    for entry in "${algorithm_models[@]}"; do
        local fname="${entry%%:*}"
        local label="${entry#*:}"
        _model_line "$PRETRAINED_DIR/$fname" "$label"
    done

    # ═══════════════════════════════════════════
    # 5. 子目录模型
    # ═══════════════════════════════════════════
    echo ""
    echo -e "  ${BLUE}▸ 子目录模型文件${NC}"
    echo ""

    local subdir_models=(
        "efficientad_pretrained_weights/pretrained_teacher_small.pth:EfficientAD 教师网络 (small)"
        "efficientad_pretrained_weights/pretrained_teacher_medium.pth:EfficientAD 教师网络 (medium)"
    )

    for entry in "${subdir_models[@]}"; do
        local fname="${entry%%:*}"
        local label="${entry#*:}"
        _model_line "$PRETRAINED_DIR/$fname" "$label"
    done

    # ═══════════════════════════════════════════
    # 6. 缺失汇总与下载指引
    # ═══════════════════════════════════════════
    if [ ${#missing_required[@]} -gt 0 ]; then
        echo ""
        echo -e "  ${YELLOW}╔══════════════════════════════════════════════════════╗${NC}"
        echo -e "  ${YELLOW}║  缺少 ${#missing_required[@]} 个关键基础模型，部分功能不可用            ║${NC}"
        echo -e "  ${YELLOW}╚══════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  ${NC}缺失文件:${NC}"
        for m in "${missing_required[@]}"; do
            echo -e "    • ${m}"
        done
        echo ""
        echo -e "  ${NC}下载地址:${NC}"
        echo -e "    DINOv2 → https://dl.fbaipublicfiles.com/dinov2/"
        echo -e "    ResNet → https://download.pytorch.org/models/"
        echo -e "    CLIP   → https://openaipublic.azureedge.net/clip/models/"
        echo ""
        echo -e "  ${NC}下载后放入: ${BLUE}${PRETRAINED_DIR}${NC}"
        echo -e "  ${NC}缺失模型将在首次运行时尝试自动下载${NC}"
    fi

    unset -f _model_line _ls_model
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
    
    # 使用固定日志文件，持续写入
    local LOG_FILE="$PROJECT_DIR/logs/backend.log"
    
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
        # 尝试回退到固定日志文件
        local fallback_log="$PROJECT_DIR/logs/backend.log"
        if [ -f "$fallback_log" ]; then
            echo -e "${BLUE}正在查看日志:${NC} $fallback_log"
            tail -f "$fallback_log"
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
