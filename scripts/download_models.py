#!/usr/bin/env python3
"""
模型下载脚本 - 自动下载项目所需的所有预训练模型
支持断点续传、完整性验证、进度反馈
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_download.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    url: str
    local_path: str
    description: str
    size_mb: float
    md5_hash: Optional[str] = None
    sha256_hash: Optional[str] = None
    source: str = "huggingface"  # huggingface, openai, github, etc.
    required: bool = True  # 是否必需
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ModelRegistry:
    """模型注册表 - 管理所有需要下载的模型"""
    
    def __init__(self, pretrained_dir: str = "pre_trained"):
        self.pretrained_dir = Path(pretrained_dir)
        self.pretrained_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelInfo] = {}
        self._register_all_models()
    
    def _register_all_models(self):
        """注册所有项目需要的模型"""
        
        # ==================== SubspaceAD / DINOv2 模型 ====================
        # 用于少样本异常检测的特征提取
        self.register(ModelInfo(
            name="dinov2-small",
            url="facebook/dinov2-small",
            local_path=str(self.pretrained_dir / "dinov2-small"),
            description="DINOv2 Small - 用于SubspaceAD少样本检测的轻量级特征提取器",
            size_mb=88.0,
            source="huggingface",
            required=True,
            dependencies=["transformers>=4.35.0", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-base",
            url="facebook/dinov2-base",
            local_path=str(self.pretrained_dir / "dinov2-base"),
            description="DINOv2 Base - 用于SubspaceAD少样本检测的基础特征提取器",
            size_mb=330.0,
            source="huggingface",
            required=True,
            dependencies=["transformers>=4.35.0", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-large",
            url="facebook/dinov2-large",
            local_path=str(self.pretrained_dir / "dinov2-large"),
            description="DINOv2 Large - 用于SubspaceAD少样本检测的高精度特征提取器",
            size_mb=1150.0,
            source="huggingface",
            required=True,
            dependencies=["transformers>=4.35.0", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-with-registers-small",
            url="facebook/dinov2-with-registers-small",
            local_path=str(self.pretrained_dir / "dinov2-with-registers-small"),
            description="DINOv2 Small with Registers - 改进版本，用于SubspaceAD",
            size_mb=88.0,
            source="huggingface",
            required=True,
            dependencies=["transformers>=4.35.0", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-with-registers-base",
            url="facebook/dinov2-with-registers-base",
            local_path=str(self.pretrained_dir / "dinov2-with-registers-base"),
            description="DINOv2 Base with Registers - 改进版本，用于SubspaceAD",
            size_mb=330.0,
            source="huggingface",
            required=True,
            dependencies=["transformers>=4.35.0", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-with-registers-large",
            url="facebook/dinov2-with-registers-large",
            local_path=str(self.pretrained_dir / "dinov2-with-registers-large"),
            description="DINOv2 Large with Registers - 改进版本，高精度特征提取器",
            size_mb=1150.0,
            source="huggingface",
            required=True,
            dependencies=["transformers>=4.35.0", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-with-registers-giant",
            url="facebook/dinov2-with-registers-giant",
            local_path=str(self.pretrained_dir / "dinov2-with-registers-giant"),
            description="DINOv2 Giant with Registers - 最高精度特征提取器",
            size_mb=2300.0,
            source="huggingface",
            required=False,
            dependencies=["transformers>=4.35.0", "torch>=2.1.0"]
        ))
        
        # ==================== MuSc / CLIP 模型 ====================
        # 用于零样本异常检测
        self.register(ModelInfo(
            name="clip-vit-b-32",
            url="openai/ViT-B-32",
            local_path=str(self.pretrained_dir / "ViT-B-32.pt"),
            description="CLIP ViT-B/32 - 用于MuSc零样本检测的基础模型",
            size_mb=340.0,
            source="open_clip",
            required=True,
            dependencies=["open_clip_torch", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="clip-vit-b-16",
            url="openai/ViT-B-16",
            local_path=str(self.pretrained_dir / "ViT-B-16.pt"),
            description="CLIP ViT-B/16 - 用于MuSc零样本检测的高精度基础模型",
            size_mb=340.0,
            source="open_clip",
            required=True,
            dependencies=["open_clip_torch", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="clip-vit-l-14",
            url="openai/ViT-L-14",
            local_path=str(self.pretrained_dir / "ViT-L-14.pt"),
            description="CLIP ViT-L/14 - 用于MuSc零样本检测的大模型",
            size_mb=890.0,
            source="open_clip",
            required=True,
            dependencies=["open_clip_torch", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="clip-vit-l-14-336",
            url="openai/ViT-L-14-336px",
            local_path=str(self.pretrained_dir / "ViT-L-14-336px.pt"),
            description="CLIP ViT-L/14@336px - 用于MuSc零样本检测的高分辨率模型",
            size_mb=890.0,
            source="open_clip",
            required=True,
            dependencies=["open_clip_torch", "torch>=2.1.0"]
        ))
        
        # ==================== Dinomaly DINOv2 模型 ====================
        # 用于Dinomaly异常检测
        self.register(ModelInfo(
            name="dinov2-vits14-pretrain",
            url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            local_path=str(self.pretrained_dir / "dinov2_vits14_pretrain.pth"),
            description="DINOv2 ViT-S/14 预训练权重 - 用于Dinomaly",
            size_mb=88.0,
            source="direct",
            required=True,
            dependencies=["torch>=2.1.0", "timm==0.9.12"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-vitb14-pretrain",
            url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
            local_path=str(self.pretrained_dir / "dinov2_vitb14_pretrain.pth"),
            description="DINOv2 ViT-B/14 预训练权重 - 用于Dinomaly",
            size_mb=330.0,
            source="direct",
            required=True,
            dependencies=["torch>=2.1.0", "timm==0.9.12"]
        ))
        
        self.register(ModelInfo(
            name="dinov2-vitl14-pretrain",
            url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            local_path=str(self.pretrained_dir / "dinov2_vitl14_pretrain.pth"),
            description="DINOv2 ViT-L/14 预训练权重 - 用于Dinomaly",
            size_mb=1150.0,
            source="direct",
            required=True,
            dependencies=["torch>=2.1.0", "timm==0.9.12"]
        ))
        
        # ==================== ResNet 骨干网络 ====================
        # 用于多种算法的骨干网络
        self.register(ModelInfo(
            name="resnet18",
            url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
            local_path=str(self.pretrained_dir / "resnet18-f37072fd.pth"),
            description="ResNet-18 预训练权重 - 用于轻量级特征提取",
            size_mb=45.0,
            source="direct",
            required=True,
            dependencies=["torch>=2.1.0", "torchvision>=0.16.0"]
        ))
        
        self.register(ModelInfo(
            name="wide-resnet50-2",
            url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
            local_path=str(self.pretrained_dir / "wide_resnet50_2-95faca4d.pth"),
            description="Wide ResNet-50-2 预训练权重 - 用于PatchCore等算法",
            size_mb=132.0,
            source="direct",
            required=True,
            dependencies=["torch>=2.1.0", "torchvision>=0.16.0"]
        ))
        
        # ==================== timm 模型 (通过timm自动下载) ====================
        # 这些模型由timm库自动管理，但我们需要记录它们
        self.register(ModelInfo(
            name="timm-resnet200",
            url="timm/resnet200",
            local_path=str(self.pretrained_dir / "timm" / "resnet200"),
            description="ResNet-200 - 用于MuSc的深层骨干网络",
            size_mb=250.0,
            source="timm",
            required=False,
            dependencies=["timm==0.9.12", "torch>=2.1.0"]
        ))
        
        self.register(ModelInfo(
            name="timm-efficientnet-b5",
            url="timm/tf_efficientnet_b5",
            local_path=str(self.pretrained_dir / "timm" / "efficientnet_b5"),
            description="EfficientNet-B5 - 用于MuSc的高效骨干网络",
            size_mb=118.0,
            source="timm",
            required=False,
            dependencies=["timm==0.9.12", "torch>=2.1.0"]
        ))
        
        logger.info(f"Registered {len(self.models)} models")
    
    def register(self, model_info: ModelInfo):
        """注册模型"""
        self.models[model_info.name] = model_info
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.models.get(name)
    
    def get_all_models(self) -> List[ModelInfo]:
        """获取所有模型"""
        return list(self.models.values())
    
    def get_required_models(self) -> List[ModelInfo]:
        """获取所有必需模型"""
        return [m for m in self.models.values() if m.required]
    
    def get_models_by_source(self, source: str) -> List[ModelInfo]:
        """按来源获取模型"""
        return [m for m in self.models.values() if m.source == source]
    
    def export_config(self, filepath: str):
        """导出模型配置到JSON"""
        config = {
            "pretrained_dir": str(self.pretrained_dir),
            "models": {name: asdict(info) for name, info in self.models.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Model configuration exported to {filepath}")


class ModelDownloader:
    """模型下载器 - 支持断点续传和完整性验证"""
    
    def __init__(self, registry: ModelRegistry, chunk_size: int = 8192):
        self.registry = registry
        self.chunk_size = chunk_size
        self.downloaded_models = []
        self.failed_models = []
    
    def _calculate_md5(self, filepath: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _calculate_sha256(self, filepath: str) -> str:
        """计算文件的SHA256哈希值"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _verify_integrity(self, filepath: str, model_info: ModelInfo) -> bool:
        """验证文件完整性"""
        if not os.path.exists(filepath):
            return False
        
        # 检查文件大小
        actual_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        if actual_size < model_info.size_mb * 0.9:  # 允许10%的误差
            logger.warning(f"File size mismatch: expected ~{model_info.size_mb}MB, got {actual_size:.1f}MB")
            return False
        
        # 如果提供了哈希值，验证哈希
        if model_info.md5_hash:
            actual_md5 = self._calculate_md5(filepath)
            if actual_md5 != model_info.md5_hash:
                logger.warning(f"MD5 mismatch: expected {model_info.md5_hash}, got {actual_md5}")
                return False
        
        if model_info.sha256_hash:
            actual_sha256 = self._calculate_sha256(filepath)
            if actual_sha256 != model_info.sha256_hash:
                logger.warning(f"SHA256 mismatch: expected {model_info.sha256_hash}, got {actual_sha256}")
                return False
        
        return True
    
    def _download_with_resume(self, url: str, destination: str, model_name: str) -> bool:
        """支持断点续传的下载"""
        try:
            import requests
            
            destination_path = Path(destination)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查已下载的部分
            downloaded_size = 0
            if destination_path.exists():
                downloaded_size = destination_path.stat().st_size
                logger.info(f"Resuming download for {model_name} from {downloaded_size} bytes")
            
            # 设置请求头支持断点续传
            headers = {}
            if downloaded_size > 0:
                headers['Range'] = f'bytes={downloaded_size}-'
            
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取总大小
            total_size = int(response.headers.get('content-length', 0))
            if 'content-range' in response.headers:
                # 处理断点续传的情况
                total_size = int(response.headers['content-range'].split('/')[-1])
            elif downloaded_size > 0:
                total_size += downloaded_size
            
            # 打开文件进行追加或新建
            mode = 'ab' if downloaded_size > 0 else 'wb'
            
            with open(destination, mode) as f:
                downloaded = downloaded_size
                start_time = time.time()
                last_update = start_time
                
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 每2秒更新一次进度
                        current_time = time.time()
                        if current_time - last_update >= 2:
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            speed = downloaded / (current_time - start_time) / 1024 / 1024  # MB/s
                            logger.info(f"{model_name}: {progress:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB) - {speed:.2f}MB/s")
                            last_update = current_time
            
            logger.info(f"Download completed: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {model_name}: {str(e)}")
            return False
    
    def _download_huggingface_model(self, model_info: ModelInfo) -> bool:
        """下载HuggingFace模型"""
        try:
            from transformers import AutoModel, AutoImageProcessor
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading HuggingFace model: {model_info.name}")
            logger.info(f"  Source: {model_info.url}")
            logger.info(f"  Destination: {model_info.local_path}")
            
            # 使用 snapshot_download 下载整个模型
            local_path = snapshot_download(
                repo_id=model_info.url,
                local_dir=model_info.local_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"HuggingFace model downloaded to: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download HuggingFace model {model_info.name}: {str(e)}")
            return False
    
    def _download_open_clip_model(self, model_info: ModelInfo) -> bool:
        """下载OpenCLIP模型"""
        try:
            import open_clip
            
            logger.info(f"Downloading OpenCLIP model: {model_info.name}")
            
            # OpenCLIP模型通过库函数下载
            model_name = model_info.url.replace("openai/", "")
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='openai')
            
            # 保存到指定路径
            torch_save_path = model_info.local_path
            Path(torch_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 获取模型状态字典并保存
            import torch
            torch.save(model.state_dict(), torch_save_path)
            
            logger.info(f"OpenCLIP model saved to: {torch_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download OpenCLIP model {model_info.name}: {str(e)}")
            return False
    
    def _download_timm_model(self, model_info: ModelInfo) -> bool:
        """下载timm模型"""
        try:
            import timm
            import torch
            
            logger.info(f"Downloading timm model: {model_info.name}")
            
            # timm模型通过create_model下载
            model_name = model_info.url.replace("timm/", "")
            model = timm.create_model(model_name, pretrained=True)
            
            # 保存到指定路径
            torch_save_path = model_info.local_path + ".pth"
            Path(torch_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), torch_save_path)
            
            logger.info(f"timm model saved to: {torch_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download timm model {model_info.name}: {str(e)}")
            return False
    
    def _download_direct_model(self, model_info: ModelInfo) -> bool:
        """直接下载模型文件"""
        logger.info(f"Downloading model: {model_info.name}")
        logger.info(f"  URL: {model_info.url}")
        logger.info(f"  Destination: {model_info.local_path}")
        
        success = self._download_with_resume(
            model_info.url,
            model_info.local_path,
            model_info.name
        )
        
        if success and self._verify_integrity(model_info.local_path, model_info):
            logger.info(f"Model {model_info.name} downloaded and verified successfully")
            return True
        elif success:
            logger.warning(f"Model {model_info.name} downloaded but integrity check failed")
            return True  # 仍然返回True，因为文件已下载
        
        return False
    
    def download_model(self, model_name: str, force: bool = False) -> bool:
        """下载单个模型"""
        model_info = self.registry.get_model(model_name)
        if not model_info:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        # 检查是否已存在
        if not force and os.path.exists(model_info.local_path):
            if self._verify_integrity(model_info.local_path, model_info):
                logger.info(f"Model {model_name} already exists and verified")
                self.downloaded_models.append(model_name)
                return True
            else:
                logger.warning(f"Model {model_name} exists but verification failed, re-downloading")
        
        # 根据来源选择下载方式
        if model_info.source == "huggingface":
            success = self._download_huggingface_model(model_info)
        elif model_info.source == "open_clip":
            success = self._download_open_clip_model(model_info)
        elif model_info.source == "timm":
            success = self._download_timm_model(model_info)
        elif model_info.source == "direct":
            success = self._download_direct_model(model_info)
        else:
            logger.error(f"Unknown source type: {model_info.source}")
            return False
        
        if success:
            self.downloaded_models.append(model_name)
        else:
            self.failed_models.append(model_name)
        
        return success
    
    def download_all(self, required_only: bool = True, force: bool = False) -> Tuple[int, int]:
        """下载所有模型"""
        if required_only:
            models = self.registry.get_required_models()
        else:
            models = self.registry.get_all_models()
        
        logger.info(f"Starting download of {len(models)} models")
        logger.info("=" * 60)
        
        success_count = 0
        failed_count = 0
        
        for i, model_info in enumerate(models, 1):
            logger.info(f"[{i}/{len(models)}] Processing: {model_info.name}")
            
            if self.download_model(model_info.name, force=force):
                success_count += 1
            else:
                failed_count += 1
            
            logger.info("-" * 60)
        
        logger.info("=" * 60)
        logger.info(f"Download Summary: {success_count} succeeded, {failed_count} failed")
        
        if self.failed_models:
            logger.warning(f"Failed models: {', '.join(self.failed_models)}")
        
        return success_count, failed_count
    
    def verify_all(self) -> Tuple[int, int]:
        """验证所有已下载模型"""
        models = self.registry.get_all_models()
        
        verified_count = 0
        failed_count = 0
        
        logger.info("Verifying all downloaded models...")
        
        for model_info in models:
            if os.path.exists(model_info.local_path):
                if self._verify_integrity(model_info.local_path, model_info):
                    logger.info(f"✓ {model_info.name}: Verified")
                    verified_count += 1
                else:
                    logger.warning(f"✗ {model_info.name}: Verification failed")
                    failed_count += 1
            else:
                logger.warning(f"✗ {model_info.name}: Not found")
                failed_count += 1
        
        logger.info(f"Verification complete: {verified_count} verified, {failed_count} failed")
        return verified_count, failed_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for ASD_for_SPK project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all required models
  python download_models.py --all
  
  # Download all models including optional ones
  python download_models.py --all --include-optional
  
  # Download specific model
  python download_models.py --model dinov2-large
  
  # Force re-download
  python download_models.py --all --force
  
  # Verify existing models
  python download_models.py --verify
  
  # Export model configuration
  python download_models.py --export-config models_config.json
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Download all required models')
    parser.add_argument('--model', type=str,
                        help='Download specific model by name')
    parser.add_argument('--include-optional', action='store_true',
                        help='Include optional models when using --all')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if model exists')
    parser.add_argument('--verify', action='store_true',
                        help='Verify all downloaded models')
    parser.add_argument('--export-config', type=str,
                        help='Export model configuration to JSON file')
    parser.add_argument('--pretrained-dir', type=str, default='pre_trained',
                        help='Directory to store downloaded models')
    
    args = parser.parse_args()
    
    # 初始化注册表
    registry = ModelRegistry(pretrained_dir=args.pretrained_dir)
    
    # 导出配置
    if args.export_config:
        registry.export_config(args.export_config)
        return
    
    # 验证模式
    if args.verify:
        downloader = ModelDownloader(registry)
        downloader.verify_all()
        return
    
    # 下载模式
    if args.all or args.model:
        downloader = ModelDownloader(registry)
        
        if args.model:
            # 下载单个模型
            success = downloader.download_model(args.model, force=args.force)
            sys.exit(0 if success else 1)
        else:
            # 下载所有模型
            success_count, failed_count = downloader.download_all(
                required_only=not args.include_optional,
                force=args.force
            )
            sys.exit(0 if failed_count == 0 else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
