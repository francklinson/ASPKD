#!/usr/bin/env python3
"""
服务器启动脚本
使用虚拟环境运行
环境配置已迁移到 start_server.sh
"""
import os
import sys
import time
import subprocess
from datetime import datetime

# ========== 颜色定义 ==========
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def print_step(step_num, total, text):
    print(f"{Colors.OKBLUE}[{step_num}/{total}] {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}  ✓ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}  ⚠ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}  ✗ {text}{Colors.ENDC}")

def print_info(text, indent=2):
    print(f"{' '*indent}ℹ {text}")

# ========== 启动时间记录 ==========
start_time = time.time()
print_header("🎵 音频异常检测系统启动")
print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")

# ========== 第1步：项目环境检查 ==========
print_step(1, 7, "检查项目环境")

project_root = os.path.dirname(os.path.abspath(__file__))
print_info(f"项目根目录: {project_root}")

# 检查必要目录
directories_to_check = [
    ("backend", "后端代码目录"),
    ("algorithms", "算法目录"),
    ("data", "数据目录"),
    ("models", "模型目录"),
    ("uploads", "上传目录"),
    ("output", "输出目录"),
]

for dir_name, description in directories_to_check:
    dir_path = os.path.join(project_root, dir_name)
    if os.path.exists(dir_path):
        print_success(f"{description}: {dir_name}/")
    else:
        print_warning(f"{description}: {dir_name}/ 不存在，将自动创建")
        os.makedirs(dir_path, exist_ok=True)

# 检查配置文件
config_path = os.path.join(project_root, "backend/config", "config.yaml")
if os.path.exists(config_path):
    print_success(f"配置文件: backend/config/config.yaml")
else:
    print_error(f"配置文件不存在: backend/config/config.yaml")
    sys.exit(1)

# ========== 第2步：加载环境配置 ==========
print_step(2, 7, "加载环境配置")

if os.path.exists(config_path):
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        env_config = config.get('environments', {})
        
        env_set_count = 0
        for key, value in env_config.items():
            if value and key not in os.environ:
                os.environ[key] = str(value)
                env_set_count += 1
                print_info(f"设置环境变量: {key}={value}")
        
        if env_set_count > 0:
            print_success(f"已设置 {env_set_count} 个环境变量")
        else:
            print_info("环境变量已设置或无需更新")
    except Exception as e:
        print_warning(f"加载环境变量失败: {e}")

# CUDA 设备检查
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    print_info("CUDA_VISIBLE_DEVICES 未设置，将暴露所有GPU")
else:
    print_info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

# ========== 第3步：检查GPU状态 ==========
print_step(3, 7, "检查GPU状态")

try:
    # 尝试获取GPU信息
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    if result.returncode == 0:
        gpus = result.stdout.strip().split('\n')
        print_success(f"检测到 {len(gpus)} 个GPU:")
        for i, gpu in enumerate(gpus):
            name, mem_total, mem_free = [x.strip() for x in gpu.split(',')]
            print_info(f"GPU {i}: {name}")
            print_info(f"      显存: {mem_free} / {mem_total}", indent=8)
    else:
        print_warning("nvidia-smi 执行失败，可能无GPU或驱动问题")
except FileNotFoundError:
    print_warning("nvidia-smi 未找到，将使用CPU模式")
except Exception as e:
    print_warning(f"GPU检测失败: {e}")

# ========== 第4步：检查Python依赖 ==========
print_step(4, 7, "检查Python依赖")

# 确保项目根目录在路径中
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print_info(f"添加项目路径: {project_root}")

dependencies = [
    ("librosa", "音频处理库", True),
    ("watchdog", "文件监控库", True),
    ("fastapi", "Web框架", True),
    ("uvicorn", "ASGI服务器", True),
    ("torch", "PyTorch深度学习框架", False),
    ("numpy", "数值计算库", True),
    ("soundfile", "音频文件读写", True),
    ("pydantic", "数据验证", True),
]

failed_deps = []
for module_name, description, required in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print_success(f"{description}: {module_name} ({version})")
    except ImportError as e:
        if required:
            print_error(f"{description}: {module_name} - 缺失")
            failed_deps.append(module_name)
        else:
            print_warning(f"{description}: {module_name} - 缺失（可选）")

if failed_deps:
    print_error(f"缺少必要依赖: {', '.join(failed_deps)}")
    print_info("请运行: pip install -r requirements.txt")
    sys.exit(1)

# ========== 第5步：检查数据库连接 ==========
print_step(5, 7, "检查数据库连接")

try:
    from backend.core.shazam.database.connector import DatabaseChecker, MySQLConnector
    from backend.core.shazam.utils.hparam import hp
    
    # 先检查并创建数据库和表
    checker = DatabaseChecker()
    checker.check_database()
    checker.check_tables()
    
    # 测试连接
    db_connector = MySQLConnector()
    print_success("MySQL数据库连接成功")
    print_info(f"数据库: {hp.fingerprint.database.database}")
    print_info(f"主机: {hp.fingerprint.database.host}:{hp.fingerprint.database.port}")
    
    # 测试查询
    try:
        db_connector.cursor.execute("SELECT COUNT(*) FROM finger_prints")
        count = db_connector.cursor.fetchone()[0]
        print_info(f"指纹库记录数: {count}")
    except Exception as e:
        print_warning(f"指纹库查询失败: {e}")
    
    db_connector.cursor.close()
    db_connector.conn.close()
    
except Exception as e:
    print_error(f"数据库连接失败: {e}")
    print_info("请检查数据库配置: backend/config/config.yaml")
    sys.exit(1)

# ========== 第6步：检查模型文件 ==========
print_step(6, 7, "检查模型文件")

model_dirs = [
    os.path.join(project_root, "models", "pre_trained"),
    os.path.join(project_root, "models", "saved"),
]

for model_dir in model_dirs:
    if os.path.exists(model_dir):
        pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if pth_files:
            print_success(f"{os.path.basename(model_dir)}: {len(pth_files)} 个模型")
            for f in pth_files[:3]:  # 只显示前3个
                print_info(f"  - {f}")
            if len(pth_files) > 3:
                print_info(f"  ... 还有 {len(pth_files)-3} 个模型")
        else:
            print_info(f"{os.path.basename(model_dir)}: 暂无模型文件")
    else:
        print_warning(f"模型目录不存在: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)

# ========== 第7步：启动服务 ==========
print_step(7, 7, "启动FastAPI服务")

# 从环境变量获取配置
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8004))

# 检查端口是否被占用
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((HOST, PORT))
    sock.close()
    
    if result == 0:
        print_error(f"端口 {PORT} 已被占用")
        print_info(f"请检查: lsof -i :{PORT}")
        print_info(f"或修改端口: export PORT=8005")
        sys.exit(1)
    else:
        print_success(f"端口 {PORT} 可用")
except Exception as e:
    print_warning(f"端口检查失败: {e}")

# 计算启动耗时
init_time = time.time() - start_time
print_success(f"初始化完成，耗时: {init_time:.2f}秒")

print_header("服务启动信息")
print(f"{Colors.OKGREEN}服务地址: http://{HOST}:{PORT}{Colors.ENDC}")
print(f"{Colors.OKGREEN}API文档:  http://{HOST}:{PORT}/docs{Colors.ENDC}")
print(f"{Colors.OKGREEN}健康检查: http://{HOST}:{PORT}/health{Colors.ENDC}")
print(f"{Colors.OKCYAN}按 Ctrl+C 停止服务{Colors.ENDC}")
print("=" * 70 + "\n")

# 启动 FastAPI
import uvicorn

uvicorn.run(
    "backend.main:app",
    host=HOST,
    port=PORT,
    reload=False,
    log_level="info"
)
