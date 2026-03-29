"""
WebSocket 管理器
用于实时推送任务进度
"""
import json
import asyncio
from typing import Dict, Set
from fastapi import WebSocket


class WebSocketManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        # task_id -> Set[WebSocket]
        self.connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> task_id
        self.socket_task_map: Dict[WebSocket, str] = {}
    
    async def handle_connection(self, websocket: WebSocket, task_id: str):
        """处理新的 WebSocket 连接"""
        await websocket.accept()
        
        # 注册连接
        if task_id not in self.connections:
            self.connections[task_id] = set()
        self.connections[task_id].add(websocket)
        self.socket_task_map[websocket] = task_id
        
        print(f"[WebSocket] 新连接: task_id={task_id}, 当前连接数: {len(self.connections[task_id])}")
        
        # 检查是否是已完成的任务，如果是，立即返回结果（仅对真实任务ID）
        if task_id not in ["broadcast", "monitor"]:
            from backend.core.task_manager import task_manager
            task_info = task_manager.get_task_result(task_id)
            if task_info and task_info["status"] in ["completed", "failed"]:
                print(f"[WebSocket] 任务 {task_id} 已完成，立即发送结果")
                await self.send_result(task_id, task_info)
        
        try:
            # 保持连接活跃
            while True:
                # 接收心跳消息
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "get_status":
                    # 支持查询任务状态（仅对真实任务ID）
                    if task_id not in ["broadcast", "monitor"]:
                        from backend.core.task_manager import task_manager
                        task_info = task_manager.get_task_result(task_id)
                        if task_info:
                            await websocket.send_json({
                                "type": "status",
                                "data": task_info
                            })
                
        except Exception as e:
            print(f"[WebSocket] 连接断开: {e}")
        
        finally:
            # 清理连接
            await self.disconnect(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        """断开连接"""
        task_id = self.socket_task_map.get(websocket)
        
        if task_id and task_id in self.connections:
            self.connections[task_id].discard(websocket)
            if not self.connections[task_id]:
                del self.connections[task_id]
        
        if websocket in self.socket_task_map:
            del self.socket_task_map[websocket]
        
        try:
            await websocket.close()
        except:
            pass
    
    async def send_progress(self, task_id: str, progress: dict):
        """发送进度更新"""
        if task_id not in self.connections:
            return
        
        disconnected = set()
        
        for websocket in self.connections[task_id]:
            try:
                await websocket.send_json({
                    "type": "progress",
                    "data": progress
                })
            except:
                disconnected.add(websocket)
        
        # 清理断开的连接
        for websocket in disconnected:
            await self.disconnect(websocket)
    
    async def send_result(self, task_id: str, result: dict):
        """发送最终结果"""
        if task_id not in self.connections:
            return
        
        disconnected = set()
        
        for websocket in self.connections[task_id]:
            try:
                await websocket.send_json({
                    "type": "result",
                    "data": result
                })
            except:
                disconnected.add(websocket)
        
        # 清理断开的连接
        for websocket in disconnected:
            await self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """广播消息给所有连接"""
        disconnected = set()
        
        for task_id, sockets in self.connections.items():
            for websocket in sockets:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.add(websocket)
        
        # 清理断开的连接
        for websocket in disconnected:
            await self.disconnect(websocket)


# 全局 WebSocket 管理器实例
websocket_manager = WebSocketManager()
