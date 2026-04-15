# 音频异常检测 - 客户端使用指南

分布式客户端监控脚本，用于监控本地目录的新增音频文件并自动上传到服务器进行检测。

## 目录

1. [功能特性](#功能特性)
2. [架构概述](#架构概述)
3. [快速开始](#快速开始)
4. [详细部署](#详细部署)
5. [配置说明](#配置说明)
6. [工作流程](#工作流程)
7. [常用命令](#常用命令)
8. [故障排除](#故障排除)
9. [生产环境部署](#生产环境部署)

---

## 功能特性

- 📁 **目录监控**：实时监控指定目录下的新增音频文件
- 📤 **自动上传**：检测到新文件后自动上传到服务器
- 🔌 **WebSocket连接**：实时接收检测结果
- 🔄 **断线重连**：网络异常时自动重连
- 📊 **心跳机制**：定期发送心跳保持在线状态
- 🛡️ **错误重试**：上传失败自动重试（最多3次）
- 📝 **完整日志**：详细的日志记录

---

## 架构概述

### 分布式架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              服务端 (Server)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FastAPI Web 服务 (backend/)                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │    │
│  │  │ /api/client │  │ /api/detection│  │ /api/monitor│  │  WebSocket │ │    │
│  │  │   客户端管理  │  │   离线检测    │  │   本地监控    │  │  实时通信   │ │    │
│  │  └──────┬──────┘  └─────────────┘  └─────────────┘  └─────┬─────┘ │    │
│  │         │                                                   │       │    │
│  │  ┌──────▼───────────────────────────────────────────────────▼─────┐ │    │
│  │  │                    ClientManager 客户端管理器                     │ │    │
│  │  │  • 客户端注册/注销  • 心跳管理  • 状态统计  • WebSocket连接       │ │    │
│  │  └─────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────▼────────────────────────────────────┐   │
│  │                     Web 前端 (frontend/index.html)                     │   │
│  │  ┌────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │   │
│  │  │ 💻 离线检测 │  │ 📡 实时监控  │  │ 🖥️ 客户端监控│  │ 📊 特征聚类   │ │   │
│  │  └────────────┘  └─────────────┘  └──────┬──────┘  └──────────────┘ │   │
│  │                                          │                           │   │
│  │                         ┌────────────────▼────────────┐              │   │
│  │                         │ 显示所有连接的客户端状态      │              │   │
│  │                         │ • 在线/离线状态              │              │   │
│  │                         │ • 上传统计                  │              │   │
│  │                         │ • 实时活动日志              │              │   │
│  │                         │ • 检测结果热力图            │              │   │
│  │                         └─────────────────────────────┘              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
           ┌────────▼────────┐  ┌──────▼──────┐  ┌───────▼───────┐
           │  客户端-01       │  │  客户端-02   │  │   客户端-03    │
           │ (client_monitor)│  │(client_monitor)│  │(client_monitor)│
           └────────┬────────┘  └──────┬──────┘  └───────┬───────┘
                    │                  │                  │
         ┌──────────▼──────────┐      │         ┌────────▼─────────┐
         │  监控目录: /audio    │      │         │ 监控目录: /wav   │
         │  • 自动检测新文件    │◄─────┘         │ • 自动检测新文件 │
         │  • HTTP上传文件      │                │ • HTTP上传文件   │
         │  • WebSocket收结果   │                │ • WebSocket收结果│
         └─────────────────────┘                └──────────────────┘
```

### 服务端 API

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/client/register` | POST | 客户端注册 |
| `/api/client/heartbeat` | POST | 客户端心跳 |
| `/api/client/status` | GET | 获取所有客户端状态 |
| `/api/client/upload` | POST | 客户端文件上传 |
| `/api/client/disconnect` | POST | 客户端断开 |
| `/api/client/ws/{client_id}` | WebSocket | 客户端WebSocket连接 |

---

## 快速开始

### 1. 服务端准备

确保服务端已启动：

```bash
cd /home/zhouchenghao/PycharmProjects/ASD_for_SPK
source .venv/bin/activate
python backend/main.py
```

访问 `http://localhost:8004` 确认服务端正常运行。

### 2. 安装客户端

**Windows (推荐方式):**
```cmd
cd client
install_windows.bat
```
该安装脚本会自动：
- 检测Python安装
- 安装pip依赖
- 创建默认配置文件
- 创建监控目录

**Windows (手动方式):**
```cmd
cd client
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

REM 创建配置文件
copy .env.example .env
```

**Linux/Mac:**
```bash
cd client
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置环境变量

复制并编辑配置文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 服务器地址（必填）
ASD_SERVER_URL=http://192.168.1.100:8004
ASD_WS_URL=ws://192.168.1.100:8004

# 客户端名称（必填，用于在服务端UI中显示）
ASD_CLIENT_NAME=生产线-A01

# 监控目录（必填）
ASD_MONITOR_DIR=/mnt/audio_files
```

### 4. 启动客户端

**Linux/Mac:**
```bash
# 使用启动脚本（推荐）
./start_client.sh start

# 或使用Python直接运行
python client_monitor.py
```

**Windows:**
```cmd
# 方法1: 使用批处理脚本 (推荐，兼容性最好)
cd client
start_client.bat start

# 方法2: 先检查环境再启动
start_client.bat setup
start_client.bat start

# 方法3: 使用PowerShell脚本
powershell -ExecutionPolicy Bypass -File start_client.ps1 start

# 方法4: 直接使用Python
python client_monitor.py
```

**Windows 批处理脚本命令:**
```cmd
start_client.bat setup   - 检查环境并安装依赖
start_client.bat start   - 启动客户端
start_client.bat stop    - 停止客户端
start_client.bat restart - 重启客户端
start_client.bat status  - 查看状态
start_client.bat test    - 测试连接
start_client.bat log     - 查看日志
```

### 5. 验证连接

**服务端查看**：
1. 打开浏览器访问 `http://server:8004`
2. 点击 `🖥️ 客户端监控` 标签
3. 确认客户端显示为"在线"状态

**客户端日志**：
```
✅ 客户端注册成功: xxxxxxxx
✅ WebSocket连接成功
👁️ 开始监控目录: /mnt/audio_files
```

### 6. 测试文件上传

```bash
# 复制音频文件到监控目录
cp test.wav /mnt/audio_files/

# 查看客户端日志
tail -f client.log
```

**预期输出**：
```
📝 检测到新文件: test.wav
📥 文件加入队列: test.wav
📤 开始上传: test.wav
✅ 上传成功: test.wav (任务: abc123...)
✅ 正常: test.wav (分数: 0.0123)
```

---

## 详细部署

### 场景1：单机测试

在同一台机器上运行服务端和客户端：

**Linux/Mac:**
```bash
# 终端1：启动服务端
cd ASD_for_SPK
python backend/main.py

# 终端2：启动客户端
cd ASD_for_SPK/client
export ASD_SERVER_URL="http://localhost:8004"
export ASD_WS_URL="ws://localhost:8004"
export ASD_MONITOR_DIR="./test_audio"
python client_monitor.py
```

**Windows:**
```cmd
REM 终端1：启动服务端（在Linux服务器上）

REM 终端2：启动客户端（Windows）
cd C:\ASD_for_SPK\client
set ASD_SERVER_URL=http://192.168.1.100:8004
set ASD_WS_URL=ws://192.168.1.100:8004
set ASD_MONITOR_DIR=C:\AudioFiles
python client_monitor.py
```

### 场景2：局域网部署

**服务端配置**：确保监听所有网卡

```python
# backend/main.py
uvicorn.run(
    "backend.main:app",
    host="0.0.0.0",  # 允许外部访问
    port=8004,
    ...
)
```

**Linux/Mac 客户端配置**（`.env`）：

```bash
ASD_SERVER_URL=http://192.168.1.100:8004
ASD_WS_URL=ws://192.168.1.100:8004
ASD_CLIENT_NAME=生产线-A01
ASD_MONITOR_DIR=/mnt/audio_files
```

**Windows 客户端配置**（`.env`）：

```bash
ASD_SERVER_URL=http://192.168.1.100:8004
ASD_WS_URL=ws://192.168.1.100:8004
ASD_CLIENT_NAME=生产线-W01
ASD_MONITOR_DIR=C:\AudioFiles
```

> **Windows 路径注意事项：**
> - 路径中可以使用 `\\` 或 `/` 作为分隔符
> - 避免在路径中使用中文或特殊字符

### 场景3：公网部署

使用 Nginx 反向代理：

```nginx
server {
    listen 80;
    server_name asd.example.com;
    
    location / {
        proxy_pass http://localhost:8004;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**客户端配置**：

```bash
ASD_SERVER_URL=https://asd.example.com
ASD_WS_URL=wss://asd.example.com
```

### 场景4：多客户端部署

重复部署步骤，在每个监控节点部署：

| 客户端 | 名称 | 监控目录 | 服务器地址 |
|--------|------|----------|------------|
| 产线A-01 | 产线A-01 | /mnt/audio/a | http://server:8004 |
| 产线A-02 | 产线A-02 | /mnt/audio/b | http://server:8004 |
| 产线B-01 | 产线B-01 | /mnt/audio/c | http://server:8004 |

---

## 配置说明

### 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `ASD_SERVER_URL` | 服务器HTTP地址 | `http://localhost:8004` |
| `ASD_WS_URL` | 服务器WebSocket地址 | `ws://localhost:8004` |
| `ASD_CLIENT_NAME` | 客户端显示名称 | `客户端-01` |
| `ASD_CLIENT_ID` | 客户端唯一ID | 自动生成 |
| `ASD_MONITOR_DIR` | 监控目录路径 | `./monitor` |
| `ASD_LOG_LEVEL` | 日志级别 | `INFO` |

### 支持的音频格式

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- AAC (.aac)
- OGG (.ogg)
- M4A (.m4a)

---

## 工作流程

### 1. 客户端启动流程

```
启动脚本
    │
    ▼
加载配置 (.env)
    │
    ▼
注册客户端 ──▶ 获取client_id
    │
    ▼
启动各组件
    │
    ├── 文件监控线程 (watchdog)
    ├── 心跳线程 (30秒间隔)
    └── WebSocket线程 (接收结果)
```

### 2. 文件上传流程

```
客户端检测到新文件
        │
        ▼
等待文件稳定（写入完成）
        │
        ▼
添加到上传队列
        │
        ▼
HTTP POST 上传文件
        │
        ├── 成功 ──▶ 更新统计 ──▶ WebSocket等待结果
        │
        └── 失败 ──▶ 重试(最多3次) ──▶ 最终失败记录错误
```

### 3. 服务端处理流程（与实时检测一致）

客户端上传的文件使用与**实时检测完全相同的处理流程**，仅在数据来源上有区别：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        服务端统一处理流程                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据来源                                                                    │
│  ├── 实时检测: 服务端本地目录监控 (backend/core/monitor_service.py)          │
│  └── 客户端检测: 客户端上传文件 (backend/core/client_detection_service.py)   │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 第1阶段: 长音频分析 (Shazam指纹识别)                                  │   │
│  │ • 使用短时傅里叶变换(STFT)提取频谱特征                                │   │
│  │ • 提取频谱峰值点(锚点)                                               │   │
│  │ • 生成音频指纹哈希值                                                 │   │
│  │ • 与参考音频库匹配定位                                               │   │
│  │                                                                     │   │
│  │ 输出: 匹配片段列表 (文件名、参考音频、时间位置)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 第2阶段: 音频切分与频谱图生成                                         │   │
│  │ • 使用 Shazam locate 精确定位                                        │   │
│  │ • 根据定位结果切分10秒片段                                           │   │
│  │ • 生成梅尔频谱图 (Mel-Spectrogram)                                   │   │
│  │                                                                     │   │
│  │ 输出: 频谱图文件路径列表                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 第3阶段: 异常检测推理                                                 │   │
│  │ • 使用选定的深度学习模型 (dinomaly/patchcore等)                      │   │
│  │ • 批量推理所有频谱图                                                 │   │
│  │ • 计算异常分数                                                       │   │
│  │                                                                     │   │
│  │ 输出: 异常分数列表 (0-1 范围)                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 第4阶段: 结果处理与推送                                               │   │
│  │ • 聚合同一文件的所有片段结果                                         │   │
│  │ • WebSocket 推送结果给所有连接的客户端                                │   │
│  │ • 前端实时显示检测结果                                               │   │
│  │                                                                     │   │
│  │ 输出: 检测结果 (文件名、异常分数、是否异常、热力图)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 两种检测方式的对比

| 特性 | 实时检测 | 客户端检测 |
|------|----------|------------|
| **数据来源** | 服务端本地目录 | 客户端上传 |
| **部署方式** | 服务端监控本地目录 | 客户端监控+上传 |
| **网络要求** | 无需网络传输 | 需要稳定的网络连接 |
| **适用场景** | 服务端本地音频来源 | 分布式音频采集点 |
| **处理流程** | ✅ 完全一致 | ✅ 完全一致 |
| **音频定位** | Shazam指纹识别 | Shazam指纹识别 |
| **检测模型** | dinomaly/patchcore | dinomaly/patchcore |
| **结果推送** | WebSocket实时推送 | WebSocket实时推送 |

### 配置统一性

服务端通过 `/api/client/config` 接口统一配置客户端检测参数：

```bash
# 查看当前配置
curl http://server:8004/api/client/config

# 更新配置（同时影响实时检测和客户端检测）
curl -X POST http://server:8004/api/client/config \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "dinomaly_dinov3_small",
    "device": "auto",
    "reference_audios": ["/path/to/ref.wav"]
  }'
```

配置项说明：
- `algorithm`: 检测算法 (dinomaly_dinov3_small, patchcore 等)
- `device`: 运行设备 (auto, cpu, cuda:0 等)
- `reference_audios`: 参考音频列表（用于 Shazam 定位）

---

## 常用命令

### 查看客户端日志

```bash
# 实时查看
tail -f client.log

# 查看上传成功记录
grep "上传成功" client.log

# 查看异常记录
grep "异常" client.log

# 统计上传数量
grep "上传成功" client.log | wc -l
```

### 重启客户端

```bash
# 找到进程并重启
pkill -f client_monitor
./start_client.sh
```

### 更新配置

```bash
# 编辑配置
vim .env

# 重启客户端
pkill -f client_monitor
./start_client.sh
```

### 复制到远程机器

```bash
# 方法1: 直接复制
scp -r client/ user@target-machine:/opt/

# 方法2: 打包传输
tar czvf client.tar.gz client/
scp client.tar.gz user@target-machine:/opt/
ssh user@target-machine "cd /opt && tar xzvf client.tar.gz"
```

---

## 检测方式选择指南

### 何时使用客户端检测？

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 音频采集点分布在不同位置 | ✅ 客户端检测 | 每个采集点部署客户端，自动上传 |
| 服务端无法直接访问音频目录 | ✅ 客户端检测 | 客户端主动推送，无需共享存储 |
| 跨网络/跨地域的音频采集 | ✅ 客户端检测 | HTTP/WebSocket 传输，支持公网部署 |
| 单台服务器本地音频 | ✅ 实时检测 | 无需网络传输，效率更高 |
| 服务端可直接访问所有音频 | ✅ 实时检测 | 配置简单，无需部署客户端 |

### 混合部署方案

大型生产环境可以同时使用两种方式：

```
┌─────────────────────────────────────────────────────────────┐
│                         服务端                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              统一处理引擎                                ││
│  │  ┌─────────────────┐  ┌─────────────────────────────┐  ││
│  │  │   实时检测模块   │  │     客户端检测模块          │  ││
│  │  │                 │  │                             │  ││
│  │  │ 本地目录监控     │  │ 接收客户端上传              │  ││
│  │  │ • /data/audio   │  │ • 客户端-A                  │  ││
│  │  │ • /mnt/recordings│  │ • 客户端-B                  │  ││
│  │  │                 │  │ • 客户端-C                  │  ││
│  │  └────────┬────────┘  └─────────────┬───────────────┘  ││
│  │           │                          │                  ││
│  │           └──────────┬───────────────┘                  ││
│  │                      ▼                                  ││
│  │         ┌──────────────────────┐                       ││
│  │         │   统一处理流程        │                       ││
│  │         │ • Shazam定位         │                       ││
│  │         │ • 频谱图生成         │                       ││
│  │         │ • 异常检测推理       │                       ││
│  │         └──────────────────────┘                       ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
  ┌─────────────┐            ┌──────────────┐
  │ 本地音频源  │            │ 远程音频采集点 │
  │             │            │              │
  │ /data/audio │            │ 生产线-A     │
  │ /mnt/record │            │ 生产线-B     │
  └─────────────┘            │ 质检站-C     │
                             └──────────────┘
```

### 性能对比

| 指标 | 实时检测 | 客户端检测 |
|------|----------|------------|
| **网络开销** | 低（无传输） | 中（文件上传） |
| **延迟** | 低（本地处理） | 中（传输+处理） |
| **并发能力** | 高 | 高（批量处理） |
| **部署复杂度** | 低 | 中 |
| **适用规模** | 单节点 | 分布式 |

---

## 故障排除

### Windows 常见问题

#### 问题: Python 命令找不到

**症状**: `'python' 不是内部或外部命令`

**解决**:
```cmd
REM 检查 Python 安装
python --version
REM 或
py --version

REM 如果未安装，从 https://python.org 下载安装
REM 安装时勾选 "Add Python to PATH"
```

#### 问题: PowerShell 执行策略限制

**症状**: `无法加载脚本，因为在此系统上禁止运行脚本`

**解决**:
```powershell
# 以管理员身份运行 PowerShell，然后执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 验证
Get-ExecutionPolicy
```

#### 问题: 路径中包含中文或特殊字符

**症状**: 监控目录无法识别或文件上传失败

**解决**:
```cmd
REM 使用英文路径
set ASD_MONITOR_DIR=C:\AudioFiles\Production

REM 避免使用
REM set ASD_MONITOR_DIR=C:\用户\音频文件\生产线一
```

### Linux/Mac 常见问题

### 问题1: 无法连接服务器

**症状**: 客户端日志显示 "注册请求异常"

**解决**:
```bash
# 检查网络连通性
ping server_ip

# 检查端口连通性 (Linux/Mac)
nc -zv server_ip 8004

# Windows 使用
telnet server_ip 8004

# 检查服务器API
curl http://server_ip:8004/health
```

### 问题2: WebSocket连接失败

**症状**: 不断重连WebSocket

**解决**:
```bash
# 检查防火墙
sudo ufw allow 8004
sudo iptables -L | grep 8004

# 测试WebSocket端点
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Host: server:8004" \
  http://server:8004/api/client/ws/test
```

### 问题3: 文件上传失败

**症状**: 上传最终失败

**解决**:
```bash
# 检查文件权限
ls -la /mnt/audio_files/

# 检查服务器磁盘空间
ssh server "df -h"

# 手动测试上传
curl -X POST \
  -F "client_id=test" \
  -F "files=@test.wav" \
  http://server:8004/api/client/upload
```

### 问题4: 性能问题

| 症状 | 原因 | 解决 |
|------|------|------|
| 上传慢 | 网络带宽不足 | 考虑压缩音频 |
| 检测慢 | GPU负载高 | 检查GPU使用率，考虑升级硬件 |
| 内存不足 | 批量处理过大 | 减小处理大小 |

---

## 生产环境部署

### 使用 systemd 服务（Linux）

创建服务文件：

```bash
sudo tee /etc/systemd/system/asd-client.service > /dev/null <<EOF
[Unit]
Description=ASD Client
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/client
EnvironmentFile=/opt/client/.env
ExecStart=/opt/client/venv/bin/python client_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

启用并启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable asd-client
sudo systemctl start asd-client
sudo systemctl status asd-client
```

### 使用 Windows 服务

#### 方法1：使用 NSSM 工具（推荐）

下载并安装 [NSSM](https://nssm.cc/download)：

```batch
REM 安装服务
nssm install ASDClient

REM 在弹出的窗口中设置：
REM Path: C:\ASD\client\venv\Scripts\python.exe
REM Startup directory: C:\ASD\client
REM Arguments: client_monitor.py
REM Environment: ASD_SERVER_URL=http://server:8004;ASD_CLIENT_NAME=产线A-01

REM 启动服务
nssm start ASDClient

REM 其他命令
nssm stop ASDClient
nssm restart ASDClient
nssm remove ASDClient
```

#### 方法2：使用 Windows 任务计划程序

创建开机启动任务：

```batch
REM 创建任务（以管理员身份运行）
schtasks /create /tn "ASD Client" /tr "C:\ASD\client\venv\Scripts\python.exe C:\ASD\client\client_monitor.py" /sc onstart /rl highest

REM 启动任务
schtasks /run /tn "ASD Client"

REM 删除任务
schtasks /delete /tn "ASD Client" /f
```

#### 方法3：创建 Windows 启动脚本

创建 `startup.bat`：

```batch
@echo off
cd /d C:\ASD\client
call venv\Scripts\activate
start /B python client_monitor.py
```

将快捷方式放入启动文件夹（`Win+R` 输入 `shell:startup`）。

### 使用 Docker 部署

**Dockerfile**：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client_monitor.py .
COPY .env .

CMD ["python", "client_monitor.py"]
```

**docker-compose.yml**：

```yaml
version: '3.8'

services:
  asd-client:
    build: .
    environment:
      - ASD_SERVER_URL=http://server:8004
      - ASD_CLIENT_NAME=产线A-01
      - ASD_MONITOR_DIR=/data/audio
    volumes:
      - /host/audio:/data/audio:ro
    restart: unless-stopped
```

### 监控告警脚本

创建检查脚本 `check_client.sh`：

```bash
#!/bin/bash
if ! pgrep -f client_monitor.py > /dev/null; then
    echo "客户端已停止，正在重启..."
    cd /opt/client && ./start_client.sh
fi
```

添加到 crontab：

```bash
*/5 * * * * /opt/client/check_client.sh
```

### 定期维护

```bash
# 清理旧日志
find . -name "*.log" -mtime +30 -delete

# 清理临时文件
find /tmp -name "asd_*" -mtime +7 -delete

# 备份配置
cp .env .env.backup.$(date +%Y%m%d)
```

### 安全配置

**防火墙规则**：

```bash
# 仅允许特定IP访问
sudo ufw allow from 192.168.1.0/24 to any port 8004
sudo ufw deny 8004
```

**Nginx 限流**：

```nginx
limit_req_zone $binary_remote_addr zone=asd:10m rate=10r/s;

server {
    location / {
        limit_req zone=asd burst=20 nodelay;
        proxy_pass http://localhost:8004;
    }
}
```

---

## 回滚方案

### 客户端回滚

```bash
# 停止客户端
sudo systemctl stop asd-client

# 恢复备份配置
cp .env.backup .env

# 启动客户端
sudo systemctl start asd-client
```

---

**许可证**：本项目仅供研究和学习使用。
