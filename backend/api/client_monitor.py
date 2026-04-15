"""
客户端监控管理 API
用于管理分布式客户端连接和文件上传
"""
import os
import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import shutil
import zipfile

from backend.core.websocket import websocket_manager

router = APIRouter()


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str
    name: str
    ip_address: str
    connected_at: datetime
    last_heartbeat: datetime
    status: str = "online"  # online, offline, error
    total_uploaded: int = 0
    anomaly_detected: int = 0
    websocket: Optional[WebSocket] = None
    current_task_id: Optional[str] = None


class ClientManager:
    """客户端连接管理器"""
    
    def __init__(self):
        self.clients: Dict[str, ClientInfo] = {}
        self.upload_tasks: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
    
    async def register_client(self, client_id: str, name: str, ip_address: str, websocket: WebSocket = None) -> ClientInfo:
        """注册新客户端或重新激活已存在的客户端"""
        async with self._lock:
            now = datetime.now()

            # 检查客户端是否已存在
            if client_id in self.clients:
                # 已存在，更新状态而不是创建新记录
                client = self.clients[client_id]
                client.name = name
                client.ip_address = ip_address
                client.status = "online"
                client.last_heartbeat = now
                client.websocket = websocket
                # 保留原有的 connected_at、total_uploaded、anomaly_detected
                print(f"[ClientManager] 客户端重新上线: {client_id} ({name}) from {ip_address}, 累计上传: {client.total_uploaded}")
            else:
                # 新客户端，创建记录
                client = ClientInfo(
                    client_id=client_id,
                    name=name,
                    ip_address=ip_address,
                    connected_at=now,
                    last_heartbeat=now,
                    websocket=websocket
                )
                self.clients[client_id] = client
                print(f"[ClientManager] 新客户端注册: {client_id} ({name}) from {ip_address}")

            return client
    
    async def unregister_client(self, client_id: str):
        """注销客户端"""
        async with self._lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                client.status = "offline"
                print(f"[ClientManager] 客户端注销: {client_id}")
                # 保持记录，只更新状态
    
    async def update_heartbeat(self, client_id: str):
        """更新心跳"""
        async with self._lock:
            if client_id in self.clients:
                self.clients[client_id].last_heartbeat = datetime.now()
                self.clients[client_id].status = "online"
    
    async def update_stats(self, client_id: str, total_uploaded: int = None, anomaly_detected: int = None):
        """更新统计信息"""
        async with self._lock:
            if client_id in self.clients:
                if total_uploaded is not None:
                    self.clients[client_id].total_uploaded = total_uploaded
                if anomaly_detected is not None:
                    self.clients[client_id].anomaly_detected = anomaly_detected
    
    def get_active_clients(self) -> List[dict]:
        """获取所有活跃客户端"""
        result = []
        now = datetime.now()
        for client in self.clients.values():
            # 检查是否超时（5分钟无心跳视为离线）
            if (now - client.last_heartbeat).total_seconds() > 300:
                client.status = "offline"
            
            result.append({
                "client_id": client.client_id,
                "name": client.name,
                "ip_address": client.ip_address,
                "status": client.status,
                "connected_at": client.connected_at.isoformat(),
                "last_heartbeat": client.last_heartbeat.isoformat(),
                "total_uploaded": client.total_uploaded,
                "anomaly_detected": client.anomaly_detected,
                "current_task": client.current_task_id
            })
        return sorted(result, key=lambda x: x["connected_at"], reverse=True)
    
    def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """获取客户端信息"""
        return self.clients.get(client_id)


# 全局客户端管理器
client_manager = ClientManager()


# ========== API 路由 ==========

class ClientRegisterRequest(BaseModel):
    """客户端注册请求"""
    client_name: str
    client_id: Optional[str] = None


class ClientHeartbeatRequest(BaseModel):
    """客户端心跳请求"""
    client_id: str
    total_uploaded: Optional[int] = None
    anomaly_detected: Optional[int] = None


class ClientStatusResponse(BaseModel):
    """客户端状态响应"""
    clients: List[dict]
    total_active: int
    total_offline: int


@router.post("/register")
async def register_client(request: ClientRegisterRequest, fastapi_request: Request):
    """
    客户端注册

    - **client_name**: 客户端名称（用于显示）
    - **client_id**: 客户端ID（可选，不传则自动生成）

    返回客户端ID和连接配置
    """
    client_id = request.client_id or str(uuid.uuid4())

    # 获取客户端真实IP地址
    try:
        # 尝试从请求头中获取真实IP（适用于反向代理场景）
        if "x-forwarded-for" in fastapi_request.headers:
            ip_address = fastapi_request.headers["x-forwarded-for"].split(",")[0].strip()
        elif "x-real-ip" in fastapi_request.headers:
            ip_address = fastapi_request.headers["x-real-ip"]
        else:
            # 直接从连接获取
            ip_address = fastapi_request.client.host if fastapi_request.client else "unknown"
    except Exception as e:
        ip_address = "unknown"
        print(f"[ClientRegister] 获取IP地址失败: {e}")

    client = await client_manager.register_client(
        client_id=client_id,
        name=request.client_name,
        ip_address=ip_address
    )

    print(f"[ClientRegister] 客户端注册成功: {client_id} ({request.client_name}) IP: {ip_address}")

    return {
        "success": True,
        "client_id": client.client_id,
        "server_time": datetime.now().isoformat(),
        "message": "客户端注册成功",
        "ip_address": ip_address
    }


@router.post("/heartbeat")
async def client_heartbeat(request: ClientHeartbeatRequest):
    """
    客户端心跳
    
    客户端应定期（建议每30秒）发送心跳以保持在线状态
    """
    await client_manager.update_heartbeat(request.client_id)
    
    if request.total_uploaded is not None or request.anomaly_detected is not None:
        await client_manager.update_stats(
            request.client_id,
            request.total_uploaded,
            request.anomaly_detected
        )
    
    return {
        "success": True,
        "server_time": datetime.now().isoformat()
    }


@router.get("/status", response_model=ClientStatusResponse)
async def get_client_status():
    """
    获取所有客户端状态
    
    供前端UI显示客户端连接情况
    """
    clients = client_manager.get_active_clients()
    
    active_count = sum(1 for c in clients if c["status"] == "online")
    offline_count = sum(1 for c in clients if c["status"] == "offline")
    
    return ClientStatusResponse(
        clients=clients,
        total_active=active_count,
        total_offline=offline_count
    )


@router.post("/upload")
async def upload_from_client(
    client_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    客户端文件上传接口
    
    使用与实时检测完全相同的处理流程：
    1. 接收客户端上传的音频文件
    2. 使用长音频分析器进行 Shazam 定位
    3. 精确定位并切分片段
    4. 批量推理检测
    5. 返回结果并通知客户端
    
    算法、设备、参考音频等配置完全由服务端全局配置决定
    
    - **client_id**: 客户端ID
    - **files**: 音频文件列表
    """
    global client_global_config
    
    # 验证客户端
    client = client_manager.get_client(client_id)
    if not client:
        raise HTTPException(status_code=401, detail="客户端未注册")
    
    # 验证文件
    if not files:
        raise HTTPException(status_code=400, detail="未上传文件")
    
    allowed_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file.filename}"
            )
    
    # 确保客户端检测服务已初始化
    from backend.core.client_monitor_service import client_detection_service
    
    if not client_detection_service._detector:
        algo = client_global_config.get("algorithm", "dinomaly_dinov3_small")
        dev = client_global_config.get("device", "auto")
        refs = client_global_config.get("reference_audios", [])
        await client_detection_service.initialize(algo, dev, refs)
    
    # 使用统一的客户端检测服务处理文件
    # 这与实时检测使用相同的处理流程：分析 -> 定位 -> 切分 -> 检测
    task_id = await client_detection_service.process_client_files(
        files=files,
        client_id=client_id,
        client_name=client.name if client else "Unknown"
    )
    
    # 更新客户端统计
    async with client_manager._lock:
        if client_id in client_manager.clients:
            client_manager.clients[client_id].current_task_id = task_id
            client_manager.clients[client_id].total_uploaded += len(files)
    
    # 获取文件名列表
    file_names = [f.filename for f in files]
    
    # 广播客户端上传事件
    await websocket_manager.broadcast({
        "type": "client_upload",
        "data": {
            "client_id": client_id,
            "client_name": client.name if client else "Unknown",
            "task_id": task_id,
            "file_count": len(files),
            "files": file_names,
            "filename": file_names[0] if file_names else "未知文件",
            "timestamp": datetime.now().isoformat()
        }
    })
    
    return {
        "success": True,
        "task_id": task_id,
        "status": "processing",
        "message": f"已接收 {len(files)} 个文件，正在使用 {client_global_config.get('algorithm', 'dinomaly_dinov3_small')} 算法处理"
    }


@router.post("/disconnect")
async def disconnect_client(client_id: str = Form(...)):
    """
    客户端断开连接
    """
    await client_manager.unregister_client(client_id)
    return {
        "success": True,
        "message": "客户端已断开"
    }


# ========== 客户端配置管理 ==========

class ClientConfigRequest(BaseModel):
    """客户端配置请求"""
    algorithm: str = "dinomaly_dinov3_small"
    device: str = "auto"
    reference_audios: List[str] = []


# 全局客户端配置（简化实现，实际应用可使用数据库）
# 注意：需要在服务端配置参考音频，否则客户端上传的文件无法匹配
client_global_config = {
    "algorithm": "dinomaly_dinov3_small",
    "device": "auto",
    "reference_audios": []
}


@router.post("/config")
async def update_client_config(request: ClientConfigRequest):
    """
    更新客户端监控配置
    
    配置将应用于所有客户端上传的文件检测
    同时更新客户端检测服务，确保与实时检测使用相同的配置
    """
    global client_global_config
    
    client_global_config["algorithm"] = request.algorithm
    client_global_config["device"] = request.device
    client_global_config["reference_audios"] = request.reference_audios
    
    # 同步更新客户端检测服务
    from backend.core.client_monitor_service import client_detection_service
    await client_detection_service.update_config(
        algorithm=request.algorithm,
        device=request.device,
        reference_audios=request.reference_audios
    )
    
    print(f"[ClientManager] 客户端配置已更新: algorithm={request.algorithm}, device={request.device}, refs={len(request.reference_audios)}")
    
    return {
        "success": True,
        "message": "配置已更新，将应用于后续所有客户端上传的文件",
        "config": client_global_config
    }


@router.get("/config")
async def get_client_config():
    """
    获取当前客户端监控配置
    """
    return {
        "success": True,
        "config": client_global_config
    }


@router.get("/detection-status")
async def get_client_detection_status():
    """
    获取客户端检测服务状态
    
    返回与实时检测相同的处理流程状态
    """
    from backend.core.client_monitor_service import client_detection_service
    
    status = client_detection_service.get_status()
    
    return {
        "success": True,
        "status": status,
        "message": "客户端检测服务使用与实时检测相同的处理流程"
    }


# ========== WebSocket 路由 ==========

@router.websocket("/ws/{client_id}")
async def client_websocket(websocket: WebSocket, client_id: str):
    """
    客户端 WebSocket 连接

    用于实时推送检测结果给特定客户端
    支持服务端重启后自动重新注册
    """
    await websocket.accept()

    # 获取客户端真实IP地址
    try:
        # 尝试从websocket.client获取IP
        client_host = websocket.client.host if websocket.client else "unknown"
    except:
        client_host = "unknown"

    # 检查客户端是否已注册（服务端重启后可能丢失）
    client = client_manager.get_client(client_id)
    if not client:
        print(f"[ClientWS] 客户端 {client_id} 未注册，等待注册消息... (IP: {client_host})")
    else:
        # 注册WebSocket连接，并更新IP地址
        async with client_manager._lock:
            client_manager.clients[client_id].websocket = websocket
            # 更新IP地址（如果之前是unknown或websocket）
            if client_manager.clients[client_id].ip_address in ("unknown", "websocket"):
                client_manager.clients[client_id].ip_address = client_host
        print(f"[ClientWS] 客户端WebSocket连接: {client_id} (IP: {client_host})")
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            
            elif message.get("type") == "register":
                # 客户端重新注册（服务端重启后）
                client_name = message.get("client_name", "未知客户端")
                # 使用WebSocket连接时获取的真实IP
                await client_manager.register_client(
                    client_id=client_id,
                    name=client_name,
                    ip_address=client_host,  # 使用从WebSocket获取的真实IP
                    websocket=websocket
                )
                await websocket.send_json({"type": "register_ack", "success": True})
                print(f"[ClientWS] 客户端通过WebSocket注册: {client_id} ({client_name}) IP: {client_host}")
            
            elif message.get("type") == "heartbeat":
                # 如果客户端不存在，返回错误提示需要重新注册
                if not client_manager.get_client(client_id):
                    await websocket.send_json({
                        "type": "heartbeat_ack",
                        "error": "CLIENT_NOT_REGISTERED",
                        "message": "客户端未注册，请重新注册"
                    })
                else:
                    await client_manager.update_heartbeat(client_id)
                    await websocket.send_json({"type": "heartbeat_ack"})
                
    except WebSocketDisconnect:
        print(f"[ClientWS] 客户端WebSocket断开: {client_id}")
    except Exception as e:
        print(f"[ClientWS] 客户端WebSocket错误 {client_id}: {e}")
    finally:
        # 清理WebSocket引用
        async with client_manager._lock:
            if client_id in client_manager.clients:
                client_manager.clients[client_id].websocket = None


# 辅助函数：向特定客户端发送消息
async def send_to_client(client_id: str, message: dict):
    """向特定客户端发送消息"""
    client = client_manager.get_client(client_id)
    if client and client.websocket:
        try:
            await client.websocket.send_json(message)
        except Exception as e:
            print(f"[ClientManager] 发送消息失败 {client_id}: {e}")


# ========== 导出功能 ==========

@router.get("/export")
async def export_client_results():
    """
    导出客户端检测结果为压缩包（包含Excel表格和热力图）
    
    与实时监控导出功能完全对齐
    """
    from backend.core.client_monitor_service import client_detection_service
    
    results = list(client_detection_service.detection_results)
    
    if not results:
        raise HTTPException(status_code=404, detail="暂无检测结果可导出")
    
    export_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_dir = os.path.join("exports", f"client_{export_id}")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # 1. 创建 Excel 文件
        try:
            import pandas as pd
            
            data = []
            for r in results:
                data.append({
                    "时间": r.get("timestamp", ""),
                    "文件名": r.get("filename", ""),
                    "客户端": r.get("client_name", ""),
                    "异常分数": r.get("anomaly_score", 0),
                    "检测结果": r.get("status", ""),
                    "是否异常": "是" if r.get("is_anomaly") else "否"
                })
            
            df = pd.DataFrame(data)
            excel_path = os.path.join(export_dir, "客户端检测结果.xlsx")
            df.to_excel(excel_path, index=False, engine='openpyxl')
        except ImportError:
            # 如果没有 pandas，使用 CSV 格式
            import csv
            csv_path = os.path.join(export_dir, "客户端检测结果.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["时间", "文件名", "客户端", "异常分数", "检测结果", "是否异常"])
                for r in results:
                    writer.writerow([
                        r.get("timestamp", ""),
                        r.get("filename", ""),
                        r.get("client_name", ""),
                        r.get("anomaly_score", 0),
                        r.get("status", ""),
                        "是" if r.get("is_anomaly") else "否"
                    ])
        
        # 2. 复制热力图到导出目录
        overlay_dir = os.path.join(export_dir, "热力图叠加原图")
        os.makedirs(overlay_dir, exist_ok=True)
        
        for r in results:
            overlay_path = r.get("overlay_path")
            if overlay_path and os.path.exists(overlay_path):
                # 使用文件名作为导出文件名
                filename = os.path.basename(overlay_path)
                export_path = os.path.join(overlay_dir, filename)
                shutil.copy2(overlay_path, export_path)
        
        # 3. 打包为 zip
        zip_filename = f"client_results_{export_id}.zip"
        zip_path = os.path.join("exports", zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
        
        # 4. 清理临时目录
        shutil.rmtree(export_dir)
        
        # 使用纯英文文件名避免编码问题
        safe_filename = f"client_{export_id}.zip"
        
        return FileResponse(
            path=zip_path,
            filename=safe_filename,
            media_type='application/zip'
        )
        
    except Exception as e:
        # 清理临时目录
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")
