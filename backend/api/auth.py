"""
用户认证和会话管理 API
"""
import os
import uuid
import time
import hashlib
from typing import Dict, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Form, Header
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 会话存储 {session_token: session_info}
SESSIONS: Dict[str, dict] = {}

# 会话过期时间（默认24小时）
SESSION_EXPIRE_SECONDS = 24 * 60 * 60


class LoginResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    username: Optional[str] = None


class SessionInfo(BaseModel):
    token: str
    username: str
    created_at: str
    expires_at: str


def generate_token(username: str) -> str:
    raw = f"{username}_{uuid.uuid4()}_{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def get_session(token: str) -> Optional[dict]:
    """获取会话信息，自动清理过期会话"""
    session = SESSIONS.get(token)
    if not session:
        return None
    if time.time() - session["created_at"] > SESSION_EXPIRE_SECONDS:
        del SESSIONS[token]
        return None
    return session


def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    """验证请求的会话令牌，未登录则拒绝"""
    if not authorization:
        raise HTTPException(status_code=401, detail="未登录，请先登录")
    token = authorization.replace("Bearer ", "")
    session = get_session(token)
    if not session:
        raise HTTPException(status_code=401, detail="会话已过期，请重新登录")
    return session


@router.post("/login", response_model=LoginResponse)
async def login(username: str = Form(...), password: str = Form(...)):
    """
    用户登录，返回会话令牌
    当前支持的用户: admin/tp123456, user1-user5/tp123456
    """
    # 验证凭证
    valid_users = {
        "admin": "tp123456",
        **{f"user{i}": "tp123456" for i in range(1, 6)}
    }

    if username not in valid_users or valid_users[username] != password:
        return LoginResponse(success=False, message="用户名或密码错误")

    # 生成令牌
    token = generate_token(username)
    now = time.time()

    SESSIONS[token] = {
        "token": token,
        "username": username,
        "created_at": now,
        "expires_at": now + SESSION_EXPIRE_SECONDS,
    }

    return LoginResponse(
        success=True,
        message="登录成功",
        token=token,
        username=username,
    )


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """用户登出，清理会话"""
    if authorization:
        token = authorization.replace("Bearer ", "")
        if token in SESSIONS:
            del SESSIONS[token]
    return {"success": True, "message": "已登出"}


@router.get("/session")
async def get_current_session(authorization: Optional[str] = Header(None)):
    """获取当前会话信息"""
    session = require_auth(authorization)
    return SessionInfo(
        token=session["token"],
        username=session["username"],
        created_at=datetime.fromtimestamp(session["created_at"]).isoformat(),
        expires_at=datetime.fromtimestamp(session["expires_at"]).isoformat(),
    )


@router.get("/sessions")
async def list_sessions():
    """列出所有活跃会话（管理用）"""
    sessions = []
    for token, session in SESSIONS.items():
        if time.time() - session["created_at"] < SESSION_EXPIRE_SECONDS:
            sessions.append({
                "username": session["username"],
                "token": token[:8] + "...",
                "created_at": datetime.fromtimestamp(session["created_at"]).isoformat(),
                "remaining_seconds": int(SESSIONS[token]["expires_at"] - time.time()),
            })
    return {"total": len(sessions), "sessions": sessions}
