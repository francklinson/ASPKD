# ASD for SPK - 音频异常检测系统

基于深度学习的音频异常检测系统，支持多种异常检测算法，提供 Web 界面和客户端监控功能。

## 项目结构

```
ASD_for_SPK/
├── backend/          # 后端服务 (FastAPI)
│   ├── api/          # API 接口
│   ├── core/         # 核心业务逻辑
│   └── config/       # 配置文件
├── client/           # 客户端监控服务
│   ├── client_monitor.py
│   └── start_client.sh
├── frontend/         # 前端页面
│   ├── index.html    # 主页面
│   ├── dataset.html  # 数据集构建页面
│   └── js/           # JavaScript 文件
├── algorithms/       # 异常检测算法库
│   ├── Dinomaly/     # Dinomaly 算法
│   ├── MuSc/         # MuSc 算法
│   ├── ADer/         # ADer 算法框架
│   └── ...           # 其他算法
├── tools/            # 工具脚本
│   └── preprocessing.py
├── logs/             # 日志文件目录
├── records/          # 工作记录目录
├── .venv/            # Python 虚拟环境
└── start_server.sh   # 服务启动脚本
```

## 快速开始

### 1. 环境准备

确保已安装 Python 3.10+ 和 CUDA（如需 GPU 支持）。

### 2. 启动后端服务

```bash
# 进入项目目录
cd /home/zhouchenghao/PycharmProjects/ASD_for_SPK

# 启动服务
./start_server.sh start

# 查看状态
./start_server.sh status

# 查看日志
./start_server.sh log
```

服务启动后访问: http://localhost:8004

### 3. 启动客户端监控（可选）

```bash
# 进入客户端目录
cd client

# 复制环境变量配置文件
cp .env.example .env

# 编辑 .env 文件，配置服务器地址和监控目录
# ASD_SERVER_URL=http://localhost:8004
# ASD_MONITOR_DIR=/path/to/monitor

# 启动客户端
bash start_client.sh start

# 查看状态
bash start_client.sh status
```

## 主要功能

### 1. 离线检测
- 上传音频文件进行异常检测
- 支持批量上传
- 显示检测结果和热力图
- 支持音频试听

### 2. 实时监控
- 监控指定目录的新文件
- 自动触发异常检测
- 实时推送检测结果

### 3. 数据集构建
- 音频文件上传和管理
- 自动切片和特征提取
- 数据集导出

### 4. 算法支持
- Dinomaly (DINOv2/DINOv3)
- MuSc
- PatchCore
- 等多种异常检测算法

## 配置说明

### 后端配置

配置文件位于 `backend/config/config.yaml`，包含：
- 服务器端口配置
- 数据库配置
- 算法参数配置
- 预处理配置

### 客户端配置

客户端配置文件位于 `client/.env`：

```bash
# 服务器地址
ASD_SERVER_URL=http://localhost:8004

# 监控目录（客户端会监控此目录下的新文件）
ASD_MONITOR_DIR=/path/to/monitor

# 客户端名称
ASD_CLIENT_NAME=客户端-01

# 日志级别
ASD_LOG_LEVEL=INFO
```

## 开发说明

### 添加新的检测算法

1. 在 `algorithms/` 目录下实现算法
2. 在 `backend/core/algorithm_registry.py` 注册算法
3. 重启服务

### API 文档

服务启动后访问: http://localhost:8004/docs

## 日志和记录

- **服务日志**: `logs/backend_YYYYMMDD_HHMMSS.log`
- **客户端日志**: `client/client.log`
- **工作记录**: `records/YYYYMMDD_*.md`

## 注意事项

1. 确保虚拟环境 `.venv` 已正确配置
2. 首次启动可能需要下载预训练模型
3. 客户端监控目录需要有读写权限
4. 建议使用 Chrome 或 Edge 浏览器访问

## 最近更新

- **2026-06-16**: 将 client 文件夹从 tools 移动到项目根目录，优化项目结构
- **2026-06-16**: 修复离线检测页面历史结果播放问题

## License

MIT License
