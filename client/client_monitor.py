#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频异常检测 - 客户端监控脚本

功能：
1. 监控指定目录下的新增音频文件
2. 自动上传文件到服务器进行检测
3. WebSocket实时接收检测结果
4. 断线重连和错误恢复机制

作者：AI Assistant
版本：1.0.0
"""

import os
import sys
import time
import json
import asyncio
import logging
import hashlib
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Set, List
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
import websockets
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent


# ============ 配置 ============

@dataclass
class ClientConfig:
    """客户端配置"""
    # 服务器配置
    server_url: str = "http://localhost:8004"  # 服务器地址
    ws_url: str = "ws://localhost:8004"        # WebSocket地址
    
    # 客户端配置
    client_name: str = "客户端-01"              # 客户端名称
    client_id: Optional[str] = None             # 客户端ID（自动生成）
    
    # 监控配置
    monitor_dir: str = "./monitor"              # 监控目录
    file_extensions: tuple = ('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a')
    
    # 重试配置
    max_retries: int = 3                        # 最大重试次数
    retry_delay: int = 5                        # 重试间隔（秒）
    upload_timeout: int = 300                   # 上传超时（秒）
    
    # 心跳配置
    heartbeat_interval: int = 30                # 心跳间隔（秒）
    
    # 日志配置
    log_level: str = "INFO"                     # 日志级别
    log_file: Optional[str] = "client.log"      # 日志文件


# ============ 日志设置 ============

def setup_logging(config: ClientConfig):
    """配置日志"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger("ClientMonitor")


# ============ 文件监控处理器 ============

class AudioFileHandler(FileSystemEventHandler):
    """音频文件事件处理器"""
    
    def __init__(self, upload_queue: asyncio.Queue, extensions: tuple, logger: logging.Logger, loop: asyncio.AbstractEventLoop):
        self.upload_queue = upload_queue
        self.extensions = extensions
        self.logger = logger
        self.loop = loop
        self.processed_files: Set[str] = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in self.extensions:
            filename = os.path.basename(file_path)
            self.logger.info(f"📝 检测到新文件: {filename}")
            
            # 使用 run_coroutine_threadsafe 在独立线程中调度协程
            asyncio.run_coroutine_threadsafe(self._queue_upload(file_path), self.loop)
    
    async def _queue_upload(self, file_path: str):
        """将文件加入上传队列"""
        # 等待文件写入完成
        await asyncio.sleep(1.0)
        
        # 检查文件是否已处理
        if file_path in self.processed_files:
            return
        
        # 等待文件稳定（确保写入完成）
        if await self._wait_for_file_stable(file_path):
            await self.upload_queue.put(file_path)
            self.processed_files.add(file_path)
            self.logger.info(f"📥 文件加入队列: {os.path.basename(file_path)}")
    
    async def _wait_for_file_stable(self, file_path: str, timeout: int = 30) -> bool:
        """等待文件稳定（写入完成）"""
        start_time = time.time()
        last_size = -1
        stable_count = 0
        
        while time.time() - start_time < timeout:
            try:
                current_size = os.path.getsize(file_path)
                
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 3:  # 连续3次大小不变认为稳定
                        return True
                else:
                    stable_count = 0
                    last_size = current_size
                
                await asyncio.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"检查文件状态时出错: {e}")
                await asyncio.sleep(0.5)
        
        self.logger.warning(f"等待文件稳定超时: {file_path}")
        return False


# ============ 客户端主类 ============

class MonitorClient:
    """监控客户端"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.logger = setup_logging(config)
        
        # 生成客户端ID
        if not config.client_id:
            self.config.client_id = self._generate_client_id()
        
        # 状态
        self.is_running = False
        self.is_registered = False
        self.ws_connected = False
        
        # 队列
        self.upload_queue: asyncio.Queue = asyncio.Queue()
        
        # 组件
        self.observer: Optional[Observer] = None
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # 统计
        self.stats = {
            "total_uploaded": 0,
            "anomaly_detected": 0,
            "upload_errors": 0,
            "last_upload_time": None
        }
        
        self.logger.info(f"🚀 客户端初始化完成: {self.config.client_name} ({self.config.client_id})")
    
    def _generate_client_id(self) -> str:
        """生成客户端ID"""
        # Windows: COMPUTERNAME, Linux/Mac: HOSTNAME
        hostname = os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or 'unknown'
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        hash_input = f"{hostname}-{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    async def start(self):
        """启动客户端"""
        self.is_running = True
        
        # 创建HTTP客户端
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.upload_timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # 注册客户端
        if not await self._register():
            self.logger.error("❌ 客户端注册失败，无法启动")
            return False
        
        # 启动所有任务
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._websocket_loop()),
            asyncio.create_task(self._upload_loop()),
            asyncio.create_task(self._file_monitor_loop())
        ]
        
        self.logger.info(f"✅ 客户端已启动，正在监控目录: {self.config.monitor_dir}")
        
        # 等待所有任务
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("客户端任务被取消")
        
        return True
    
    async def stop(self, timeout: float = 10.0):
        """停止客户端（带超时保护）"""
        self.logger.info(f"🛑 正在停止客户端（超时: {timeout}秒）...")
        self.is_running = False

        try:
            # 使用超时保护整个停止过程
            await asyncio.wait_for(self._do_stop(), timeout=timeout)
            self.logger.info("👋 客户端已正常停止")
        except asyncio.TimeoutError:
            self.logger.warning("⏱️ 停止超时，执行强制清理...")
            await self._force_stop()

    async def _do_stop(self):
        """执行优雅的停止流程"""
        # 1. 先通知服务器断开（WebSocket还连着时通知更可靠）
        if self.is_registered and self.http_client:
            try:
                await asyncio.wait_for(
                    self._notify_disconnect(),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("通知服务器断开超时")
            except Exception as e:
                self.logger.warning(f"通知服务器断开时出错: {e}")

        # 2. 断开WebSocket
        if self.ws_connection:
            try:
                # 发送关闭帧，优雅关闭
                await asyncio.wait_for(
                    self.ws_connection.close(),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("WebSocket关闭超时")
            except:
                pass
            finally:
                self.ws_connection = None

        # 3. 停止文件监控（带超时）
        if self.observer:
            try:
                self.observer.stop()
                # 使用线程join的超时版本
                import threading
                if threading.current_thread() != self.observer:
                    self.observer.join(timeout=3.0)
                    if self.observer.is_alive():
                        self.logger.warning("文件监控线程未正常停止")
            except Exception as e:
                self.logger.warning(f"停止文件监控时出错: {e}")

        # 4. 关闭HTTP客户端
        if self.http_client:
            try:
                await asyncio.wait_for(
                    self.http_client.aclose(),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("HTTP客户端关闭超时")
            except:
                pass

    async def _notify_disconnect(self):
        """通知服务器客户端断开（异步发送，不阻塞）"""
        try:
            response = await self.http_client.post(
                urljoin(self.config.server_url, "/api/client/disconnect"),
                data={"client_id": self.config.client_id},
                timeout=2.0  # 短超时，快速失败
            )
            if response.status_code == 200:
                self.logger.debug("已通知服务器断开连接")
        except Exception:
            # 静默失败，不影响停止流程
            pass

    async def _force_stop(self):
        """强制停止（清理资源，不保证通知服务器）"""
        self.logger.warning("执行强制停止...")

        # 强制关闭WebSocket
        if self.ws_connection:
            try:
                await self.ws_connection.close()
            except:
                pass
            self.ws_connection = None

        # 强制停止文件监控
        if self.observer:
            try:
                self.observer.stop()
            except:
                pass

        # 强制关闭HTTP客户端
        if self.http_client:
            try:
                await self.http_client.aclose()
            except:
                pass

        self.logger.info("👋 客户端已强制停止")
    
    async def _register(self) -> bool:
        """注册客户端到服务器"""
        try:
            response = await self.http_client.post(
                urljoin(self.config.server_url, "/api/client/register"),
                json={
                    "client_name": self.config.client_name,
                    "client_id": self.config.client_id
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.is_registered = True
                    self.logger.info(f"✅ 客户端注册成功: {data.get('client_id')}")
                    return True
            
            self.logger.error(f"❌ 注册失败: {response.text}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 注册请求异常: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.is_running:
            try:
                if self.is_registered:
                    response = await self.http_client.post(
                        urljoin(self.config.server_url, "/api/client/heartbeat"),
                        json={
                            "client_id": self.config.client_id,
                            "total_uploaded": self.stats["total_uploaded"],
                            "anomaly_detected": self.stats["anomaly_detected"]
                        }
                    )
                    
                    if response.status_code != 200:
                        self.logger.warning(f"心跳发送失败: {response.status_code}")
                        # 如果心跳失败，可能是服务端重启了，尝试重新注册
                        if response.status_code == 404 or response.status_code == 401:
                            self.logger.warning("⚠️ 客户端在服务端不存在，尝试重新注册...")
                            self.is_registered = False
                            await self._register()
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self.logger.warning(f"心跳异常: {e}")
                await asyncio.sleep(5)
    
    async def _websocket_loop(self):
        """WebSocket连接循环（带自动重连）"""
        while self.is_running:
            try:
                ws_url = f"{self.config.ws_url}/api/client/ws/{self.config.client_id}"
                
                self.logger.info(f"🔌 正在连接WebSocket: {ws_url}")
                
                async with websockets.connect(ws_url) as ws:
                    self.ws_connection = ws
                    self.ws_connected = True
                    self.logger.info("✅ WebSocket连接成功")
                    
                    # 发送注册消息（服务端重启后需要重新注册）
                    await ws.send(json.dumps({
                        "type": "register",
                        "client_name": self.config.client_name
                    }))
                    
                    # 接收消息循环
                    while self.is_running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=60)
                            data = json.loads(message)
                            await self._handle_ws_message(data)
                        except asyncio.TimeoutError:
                            # 发送ping保持连接
                            await ws.send(json.dumps({"type": "ping"}))
                        except websockets.exceptions.ConnectionClosed:
                            break
                
            except Exception as e:
                self.logger.warning(f"WebSocket异常: {e}")
            
            finally:
                self.ws_connected = False
                self.ws_connection = None
            
            if self.is_running:
                self.logger.info(f"🔄 {self.config.retry_delay}秒后重连WebSocket...")
                await asyncio.sleep(self.config.retry_delay)
    
    async def _handle_ws_message(self, data: dict):
        """处理WebSocket消息"""
        msg_type = data.get("type")
        
        if msg_type == "pong":
            pass  # 心跳响应
        elif msg_type == "register_ack":
            # 注册确认
            if data.get("success"):
                self.logger.info("✅ WebSocket注册成功")
            else:
                self.logger.warning(f"⚠️ WebSocket注册失败: {data.get('message', '未知错误')}")
        elif msg_type == "heartbeat_ack":
            # 心跳确认
            if data.get("error") == "CLIENT_NOT_REGISTERED":
                self.logger.warning("⚠️ 服务端未识别客户端，将通过WebSocket重新注册")
                # 下次WebSocket重连时会自动发送注册消息
        elif msg_type == "progress":
            # 任务进度更新
            progress_data = data.get("data", {})
            progress = progress_data.get("progress", 0)
            message = progress_data.get("message", "")
            self.logger.info(f"📊 任务进度: {progress}% - {message}")
        elif msg_type == "result":
            # 检测结果
            result_data = data.get("data", {})
            await self._handle_detection_result(result_data)
        else:
            self.logger.debug(f"收到消息: {data}")
    
    async def _handle_detection_result(self, result: dict):
        """处理检测结果"""
        results = result.get("results", [])
        
        for item in results:
            filename = item.get("filename", "unknown")
            is_anomaly = item.get("is_anomaly", False)
            score = item.get("anomaly_score", 0)
            
            if is_anomaly:
                self.stats["anomaly_detected"] += 1
                self.logger.warning(f"⚠️ 异常检测: {filename} (分数: {score:.4f})")
            else:
                self.logger.info(f"✅ 正常: {filename} (分数: {score:.4f})")
    
    async def _upload_loop(self):
        """文件上传循环"""
        while self.is_running:
            try:
                # 从队列获取文件
                file_path = await asyncio.wait_for(self.upload_queue.get(), timeout=1.0)
                
                # 上传文件
                await self._upload_file(file_path)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"上传循环异常: {e}")
    
    async def _upload_file(self, file_path: str, retry_count: int = 0):
        """上传单个文件（带重试）"""
        filename = os.path.basename(file_path)
        
        try:
            self.logger.info(f"📤 开始上传: {filename}")
            
            # 准备文件 - 只上传文件和客户端ID，算法/设备由服务端决定
            with open(file_path, 'rb') as f:
                files = {'files': (filename, f, 'application/octet-stream')}
                data = {
                    'client_id': self.config.client_id
                }
                
                # 发送请求
                response = await self.http_client.post(
                    urljoin(self.config.server_url, "/api/client/upload"),
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    self.stats["total_uploaded"] += 1
                    self.stats["last_upload_time"] = datetime.now().isoformat()
                    self.logger.info(f"✅ 上传成功: {filename} (任务: {result.get('task_id', 'N/A')[:8]}...)")
                    return True
                else:
                    raise Exception(result.get("message", "上传失败"))
            elif response.status_code == 401:
                # 客户端未注册，需要重新注册
                self.logger.warning(f"⚠️ 客户端未注册，尝试重新注册...")
                self.is_registered = False
                if await self._register():
                    # 重新上传
                    return await self._upload_file(file_path, retry_count)
                else:
                    raise Exception("重新注册失败")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.error(f"❌ 上传失败 ({filename}): {e}")
            
            # 重试逻辑
            if retry_count < self.config.max_retries:
                retry_count += 1
                self.logger.info(f"🔄 {retry_count}/{self.config.max_retries}秒后重试上传: {filename}...")
                await asyncio.sleep(self.config.retry_delay * retry_count)
                return await self._upload_file(file_path, retry_count)
            else:
                self.stats["upload_errors"] += 1
                self.logger.error(f"💥 上传最终失败: {filename}")
                return False
    
    async def _file_monitor_loop(self):
        """文件监控循环"""
        # 确保监控目录存在
        os.makedirs(self.config.monitor_dir, exist_ok=True)
        
        # 创建事件处理器
        event_handler = AudioFileHandler(
            upload_queue=self.upload_queue,
            extensions=self.config.file_extensions,
            logger=self.logger,
            loop=asyncio.get_running_loop()
        )
        
        # 创建观察者
        self.observer = Observer()
        self.observer.schedule(event_handler, self.config.monitor_dir, recursive=False)
        self.observer.start()
        
        self.logger.info(f"👁️ 开始监控目录: {self.config.monitor_dir}")
        
        # 保持运行
        while self.is_running:
            await asyncio.sleep(1)
        
        self.observer.stop()
        self.observer.join()


# ============ 主函数 ============

def load_config_from_env() -> ClientConfig:
    """从环境变量和.env文件加载配置"""
    # 尝试加载 .env 文件
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✅ 已加载配置文件: {env_path}")
    except ImportError:
        pass  # python-dotenv 未安装时忽略
    
    config = ClientConfig()
    
    if os.getenv('ASD_SERVER_URL'):
        config.server_url = os.getenv('ASD_SERVER_URL')
    if os.getenv('ASD_WS_URL'):
        config.ws_url = os.getenv('ASD_WS_URL')
    if os.getenv('ASD_CLIENT_NAME'):
        config.client_name = os.getenv('ASD_CLIENT_NAME')
    if os.getenv('ASD_CLIENT_ID'):
        config.client_id = os.getenv('ASD_CLIENT_ID')
    if os.getenv('ASD_MONITOR_DIR'):
        config.monitor_dir = os.getenv('ASD_MONITOR_DIR')
    if os.getenv('ASD_LOG_LEVEL'):
        config.log_level = os.getenv('ASD_LOG_LEVEL')
    if os.getenv('ASD_LOG_FILE'):
        config.log_file = os.getenv('ASD_LOG_FILE')
    
    return config


async def main():
    """主函数"""
    # 加载配置
    config = load_config_from_env()
    
    # 打印启动信息
    print("=" * 60)
    print("🔊 音频异常检测 - 客户端监控脚本")
    print("=" * 60)
    print(f"客户端名称: {config.client_name}")
    print(f"服务器地址: {config.server_url}")
    print(f"监控目录: {config.monitor_dir}")
    print("=" * 60)
    
    # 创建客户端
    client = MonitorClient(config)
    
    # 设置信号处理（Windows 只支持 SIGINT）
    stop_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("\n🛑 收到中断信号，正在停止...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # Windows 不支持 SIGTERM
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # 启动客户端
    start_task = asyncio.create_task(client.start())
    stop_task = asyncio.create_task(stop_event.wait())

    # 等待停止信号或客户端异常
    try:
        done, pending = await asyncio.wait(
            [start_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # 取消 pending 的任务
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"客户端异常: {e}")
    finally:
        # 取消启动任务（如果还在运行）
        if not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass

        # 执行停止（带超时）
        await client.stop(timeout=10.0)


if __name__ == "__main__":
    asyncio.run(main())
