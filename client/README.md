# 音频异常检测 - 客户端监控脚本

分布式客户端监控脚本，用于监控本地目录的新增音频文件并自动上传到服务器进行检测。

## 功能特性

- 📁 **目录监控**：实时监控指定目录下的新增音频文件
- 📤 **自动上传**：检测到新文件后自动上传到服务器
- 🔌 **WebSocket连接**：实时接收检测结果
- 🔄 **断线重连**：网络异常时自动重连
- 📊 **心跳机制**：定期发送心跳保持在线状态
- 🛡️ **错误重试**：上传失败自动重试（最多3次）
- 📝 **完整日志**：详细的日志记录

## 安装

### 1. 创建虚拟环境（推荐）

```bash
cd client
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python client_monitor.py
```

### 使用环境变量配置

```bash
# 设置服务器地址
export ASD_SERVER_URL="http://192.168.1.100:8004"
export ASD_WS_URL="ws://192.168.1.100:8004"

# 设置客户端信息
export ASD_CLIENT_NAME="生产线-01"
export ASD_CLIENT_ID="client-001"

# 设置监控目录
export ASD_MONITOR_DIR="/path/to/audio/files"

# 设置检测算法
export ASD_ALGORITHM="dinomaly_dinov3_small"
export ASD_DEVICE="auto"

# 运行
python client_monitor.py
```

### Windows批处理脚本示例

创建 `start_client.bat`：

```batch
@echo off
set ASD_SERVER_URL=http://192.168.1.100:8004
set ASD_WS_URL=ws://192.168.1.100:8004
set ASD_CLIENT_NAME=生产线-01
set ASD_MONITOR_DIR=D:\AudioFiles
set ASD_ALGORITHM=dinomaly_dinov3_small

python client_monitor.py
pause
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `ASD_SERVER_URL` | 服务器HTTP地址 | `http://localhost:8004` |
| `ASD_WS_URL` | 服务器WebSocket地址 | `ws://localhost:8004` |
| `ASD_CLIENT_NAME` | 客户端显示名称 | `客户端-01` |
| `ASD_CLIENT_ID` | 客户端唯一ID | 自动生成 |
| `ASD_MONITOR_DIR` | 监控目录路径 | `./monitor` |
| `ASD_ALGORITHM` | 检测算法 | `dinomaly_dinov3_small` |
| `ASD_DEVICE` | 运行设备 | `auto` |
| `ASD_LOG_LEVEL` | 日志级别 | `INFO` |

## 支持的音频格式

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- AAC (.aac)
- OGG (.ogg)
- M4A (.m4a)

## 工作原理

1. **文件监控**：使用 `watchdog` 库监控指定目录
2. **文件稳定检测**：等待文件写入完成后才上传
3. **HTTP上传**：通过HTTP POST上传文件到服务器
4. **WebSocket连接**：建立WebSocket连接接收实时结果
5. **心跳维持**：每30秒发送心跳保持在线状态

## 日志查看

日志默认输出到控制台和 `client.log` 文件。

```bash
# 实时查看日志
tail -f client.log

# Windows
Get-Content client.log -Wait
```

## 故障排除

### 无法连接服务器

- 检查服务器地址是否正确
- 确认防火墙允许连接
- 检查服务器是否正常运行

### 文件上传失败

- 检查文件是否被其他程序占用
- 查看日志中的错误信息
- 确认网络连接稳定

### WebSocket断线

- 客户端会自动重连（最多重试3次）
- 检查服务器WebSocket服务是否正常

## 网络拓扑支持

本客户端支持以下网络环境：

- **局域网**：服务器和客户端在同一内网
- **跨网段**：通过路由器转发
- **VPN环境**：通过VPN隧道连接
- **公网部署**：服务器有公网IP或使用内网穿透

## 许可证

本项目仅供研究和学习使用。
