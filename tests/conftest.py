"""
测试公共 fixtrues 和配置
"""
import os
import sys
import json
import wave
import struct
import io
import tempfile
import pytest
import pytest_asyncio
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


# ==================== HTTP 客户端 ====================

@pytest_asyncio.fixture
async def client():
    """创建异步 HTTP 测试客户端"""
    from httpx import ASGITransport, AsyncClient
    from backend.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sync_client():
    """创建同步 HTTP 测试客户端"""
    from httpx import Client, ASGITransport
    from backend.main import app

    transport = ASGITransport(app=app)
    with Client(transport=transport, base_url="http://test") as sc:
        yield sc


# ==================== Auth Fixture ====================

@pytest_asyncio.fixture
async def auth_token(client):
    """获取管理员登录 token"""
    resp = await client.post("/api/auth/login", data={"username": "admin", "password": "tp123456"})
    data = resp.json()
    return data["token"]


@pytest_asyncio.fixture
async def auth_headers(auth_token):
    """获取带认证的请求头"""
    return {"Authorization": f"Bearer {auth_token}"}


# ==================== 测试数据生成 ====================

def _create_test_wav(filepath: str, duration_sec: float = 1.0, sample_rate: int = 22050) -> str:
    """生成一个简单的测试 WAV 文件（440Hz 正弦波）"""
    num_samples = int(duration_sec * sample_rate)
    frequency = 440.0
    amplitude = 16000

    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(amplitude * __import__('math').sin(2 * __import__('math').pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack('<h', value))
    return filepath


def _create_test_image(filepath: str, size: tuple = (224, 224)) -> str:
    """生成一个简单的测试 PNG 图像"""
    try:
        from PIL import Image
        img = Image.new('RGB', size, color=(73, 109, 137))
        img.save(filepath)
    except ImportError:
        # 如果 PIL 不可用，创建一个最小的有效PNG
        import base64
        # 最小PNG (1x1红色像素)
        minimal_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        from PIL import Image
        import io as io_module
        img = Image.open(io_module.BytesIO(minimal_png))
        img = img.resize(size)
        img.save(filepath)
    return filepath


@pytest.fixture(scope="session")
def test_wav_file():
    """创建测试用 WAV 文件（session 级别）"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    filepath = os.path.join(TEST_DATA_DIR, "test_audio.wav")
    return _create_test_wav(filepath)


@pytest.fixture(scope="session")
def test_wav_files():
    """创建多个测试 WAV 文件"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    files = []
    for i in range(3):
        filepath = os.path.join(TEST_DATA_DIR, f"test_audio_{i}.wav")
        _create_test_wav(filepath, duration_sec=1.0 + i * 0.5)
        files.append(filepath)
    return files


@pytest.fixture(scope="session")
def test_image_file():
    """创建测试用 PNG 图像文件"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    filepath = os.path.join(TEST_DATA_DIR, "test_image.png")
    return _create_test_image(filepath)


@pytest.fixture(scope="session")
def test_image_files():
    """创建多个测试 PNG 图像文件"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    files = []
    for i in range(8):
        filepath = os.path.join(TEST_DATA_DIR, f"test_image_{i}.png")
        _create_test_image(filepath)
        files.append(filepath)
    return files


@pytest.fixture(scope="session")
def test_mp3_file():
    """创建测试用 MP3 文件"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    filepath = os.path.join(TEST_DATA_DIR, "test_audio.mp3")
    # 使用 pydub 或直接写入 wav 改扩展名（仅用于测试格式验证）
    _create_test_wav(filepath)
    return filepath


def generate_invalid_file(filepath: str):
    """生成一个无效的文本文件用于格式验证测试"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("This is not an audio file")
    return filepath
