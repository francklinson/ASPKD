"""
MuSc算法适配器 - 完整版
Zero-Shot工业异常检测 - ICLR 2024
还原核心流程：LNAMD + MSM + RsCIN
"""

import os
import sys
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image
import cv2

from core import BaseDetector, DetectionResult, register_algorithm


class MuScBaseAdapter(BaseDetector):
    """MuSc基础适配器 - 完整 LNAMD + MSM + RsCIN 流程"""
    
    def __init__(self, model_path: str, device: str = 'auto', threshold: float = 0.5,
                 backbone: str = 'ViT-L-14-336', img_size: int = 336, **kwargs):
        super().__init__(model_path, device, threshold, **kwargs)
        self.backbone = backbone
        self.img_size = img_size
        self._musc = None
        self._preprocess = None
        self.config = kwargs
        
        # MuSc特定参数
        self.r_list = self.config.get('r_list', [1, 3, 5])
        self.divide_num = self.config.get('divide_num', 1)
        self.batch_size = self.config.get('batch_size', 1)
        self.vis = self.config.get('vis', False)
        self.vis_type = self.config.get('vis_type', 'single_norm')
        
        # 参考集用于单图推理（模拟正常样本）
        self._reference_features = None
        self._reference_cls_tokens = None
        
    def load_model(self) -> None:
        """加载MuSc模型"""
        import time
        start_time = time.time()
        
        print(f"[MuSc] {'='*60}")
        print(f"[MuSc] [MODEL LOAD START] MuSc Zero-Shot Detector")
        print(f"[MuSc] {'='*60}")
        print(f"[MuSc] Configuration:")
        print(f"[MuSc]   - Backbone: {self.backbone}")
        print(f"[MuSc]   - Image size: {self.img_size}x{self.img_size}")
        print(f"[MuSc]   - Device: {self.device}")
        print(f"[MuSc]   - R list (scale factors): {self.r_list}")
        print(f"[MuSc]   - Divide num: {self.divide_num}")
        
        # 添加MuSc目录到路径
        musc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MuSc')
        if musc_dir not in sys.path:
            sys.path.insert(0, musc_dir)
            print(f"[MuSc] Added to sys.path: {musc_dir}")
        
        # 导入MuSc相关模块
        print(f"[MuSc] [1/2] Importing MuSc modules...")
        try:
            from MuSc.models.musc import MuSc as MuScModel
            from MuSc.models.modules.LNAMD import LNAMD
            from MuSc.models.modules.MSM import MSM
            from MuSc.models.modules.RsCIN import RsCIN
            print(f"[MuSc] ✓ Modules imported successfully")
        except Exception as e:
            print(f"[MuSc] ✗ Failed to import MuSc modules: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 创建配置字典
        device_str = str(self.device)
        
        if device_str == 'cpu':
            device_cfg = 'cpu'
        elif device_str.startswith('cuda'):
            if ':' in device_str:
                device_cfg = device_str.split(':')[1]
            else:
                device_cfg = '0'
        else:
            device_cfg = '0'
        
        # 从配置中读取MuSc配置
        from core import ConfigManager
        config_manager = ConfigManager()
        
        # 默认配置
        pretrained_models_dir = 'pre_trained'
        feature_layers = [5, 11, 17, 23]
        pretrained = 'openai' if not self.backbone.startswith('dino') else 'laion400m_e31'
        
        # 尝试从配置读取
        try:
            if hasattr(config_manager, 'config') and 'models' in config_manager.config:
                models_config = config_manager.config['models']
                if 'musc' in models_config:
                    musc_config = models_config['musc']
                    if isinstance(musc_config, dict):
                        if 'pretrained_models_dir' in musc_config:
                            pretrained_models_dir = musc_config['pretrained_models_dir']
                        if 'backbones' in musc_config and self.backbone in musc_config['backbones']:
                            backbone_config = musc_config['backbones'][self.backbone]
                            if 'feature_layers' in backbone_config:
                                feature_layers = backbone_config['feature_layers']
                            if 'pretrained' in backbone_config:
                                pretrained = backbone_config['pretrained']
        except Exception as e:
            print(f"[MuSc] Warning: Failed to load config, using defaults: {e}")
        
        print(f"[MuSc] Model configuration:")
        print(f"[MuSc]   - Pretrained models dir: {pretrained_models_dir}")
        print(f"[MuSc]   - Feature layers: {feature_layers}")
        print(f"[MuSc]   - Pretrained source: {pretrained}")
        print(f"[MuSc]   - Device cfg: {device_cfg}")
        
        cfg = {
            'device': device_cfg,
            'datasets': {
                'data_path': os.path.dirname(self.model_path),
                'dataset_name': 'mvtec_ad',
                'class_name': 'all',
                'divide_num': self.divide_num
            },
            'models': {
                'pretrained_models_dir': pretrained_models_dir,
                'backbone_name': self.backbone,
                self.backbone: {
                    'img_resize': self.img_size,
                    'feature_layers': feature_layers,
                    'pretrained': pretrained
                },
                'batch_size': self.batch_size,
                'r_list': self.r_list
            },
            'testing': {
                'output_dir': './output',
                'vis': self.vis,
                'vis_type': self.vis_type,
                'save_excel': False
            }
        }
        
        # 创建MuSc模型实例
        print(f"[MuSc] [2/2] Creating MuSc model instance...")
        print(f"[MuSc]   - Loading {self.backbone} backbone (this may take a while)...")
        
        try:
            model_create_start = time.time()
            self._musc = MuScModel(cfg, seed=42)
            model_create_time = time.time() - model_create_start
            print(f"[MuSc] ✓ MuSc model created in {model_create_time:.2f}s")
        except Exception as e:
            print(f"[MuSc] ✗ Failed to create MuSc model: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 存储配置
        self.feature_layers = [l + 1 for l in feature_layers]
        
        self.is_loaded = True
        total_time = time.time() - start_time
        print(f"[MuSc] {'='*60}")
        print(f"[MuSc] [MODEL LOAD COMPLETE] Total time: {total_time:.2f}s")
        print(f"[MuSc] {'='*60}")
    
    def _extract_features(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        提取图像特征
        
        Args:
            images: [B, 3, H, W] 图像张量
            
        Returns:
            patch_tokens: 列表，每个元素是 [B, num_patches+1, dim] 的张量
            image_features: [B, dim] 图像级特征
        """
        with torch.no_grad():
            if 'dinov2' in self.backbone.lower():
                # DINOv2
                patch_tokens_all = self._musc.dino_model.get_intermediate_layers(
                    x=images,
                    n=[l - 1 for l in self.feature_layers],
                    return_class_token=False
                )
                image_features = self._musc.dino_model(images)
                patch_tokens = [patch_tokens_all[l].cpu() for l in range(len(self.feature_layers))]
                # 添加 fake class token
                fake_cls = [torch.zeros_like(p)[:, 0:1, :] for p in patch_tokens]
                patch_tokens = [torch.cat([fake_cls[i], patch_tokens[i]], dim=1) for i in range(len(patch_tokens))]
            elif 'dino' in self.backbone.lower():
                # DINO
                patch_tokens_all = self._musc.dino_model.get_intermediate_layers(
                    x=images,
                    n=max(self.feature_layers)
                )
                image_features = self._musc.dino_model(images)
                patch_tokens = [patch_tokens_all[l - 1].cpu() for l in self.feature_layers]
            else:
                # CLIP
                image_features, patch_tokens = self._musc.clip_model.encode_image(images, self.feature_layers)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.feature_layers))]
        
        return patch_tokens, image_features
    
    def _run_lnamd_msm(self, patch_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        执行完整的 LNAMD + MSM 流程
        
        注意：不同 r 值会产生不同数量的 patch，因此每个 r 值处理后
        需要立即插值到图像尺寸，然后再合并所有 r 值的结果
        
        Args:
            patch_tokens: 列表，每个元素是 [B, num_patches+1, dim]
            
        Returns:
            anomaly_maps: [B, img_size, img_size] 异常热力图
        """
        # 添加MuSc目录到路径
        musc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MuSc')
        if musc_dir not in sys.path:
            sys.path.insert(0, musc_dir)
        
        from MuSc.models.modules.LNAMD import LNAMD
        from MuSc.models.modules.MSM import MSM
        
        feature_dim = patch_tokens[0][0].shape[-1]
        batch_size = patch_tokens[0].shape[0]
        
        # 每个 r 值处理后立即插值到图像尺寸，避免 patch 数量不匹配
        anomaly_maps_r_resized = []
        
        for r in self.r_list:
            # LNAMD 处理
            lnamd = LNAMD(compute_device=self.device, r=r, feature_dim=feature_dim, 
                         feature_layer=self.feature_layers)
            
            Z_layers = {}
            for batch_idx in range(batch_size):
                current_tokens = [p[batch_idx:batch_idx+1].to(self.device) for p in patch_tokens]
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    features = lnamd._embed(current_tokens)
                    features = features / features.norm(dim=-1, keepdim=True)
                    
                    for l in range(len(self.feature_layers)):
                        if str(l) not in Z_layers:
                            Z_layers[str(l)] = []
                        Z_layers[str(l)].append(features[:, :, l, :])
            
            # MSM 处理 - 对每个 layer
            anomaly_maps_l_list = []
            
            for l in Z_layers.keys():
                Z = torch.cat(Z_layers[l], dim=0).to(self.device)  # [B, num_patches, C]
                
                # 使用完整 MSM
                anomaly_maps_msm = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3)
                anomaly_maps_l_list.append(anomaly_maps_msm.unsqueeze(0).cpu())
            
            # 平均多个 layer 的结果
            anomaly_maps_l = torch.mean(torch.cat(anomaly_maps_l_list, dim=0), dim=0).to(self.device)  # [B, L]
            
            # 立即插值到图像尺寸（不同 r 值的 L 不同，不能直接合并）
            B, L = anomaly_maps_l.shape
            H = int(np.sqrt(L))
            anomaly_maps_resized = F.interpolate(
                anomaly_maps_l.view(B, 1, H, H),
                size=self.img_size,
                mode='bilinear',
                align_corners=True
            ).squeeze(1).cpu()  # [B, img_size, img_size]
            
            anomaly_maps_r_resized.append(anomaly_maps_resized.unsqueeze(0))  # [1, B, img_size, img_size]
        
        # 合并所有 r 值的结果（现在都在图像空间，尺寸一致）
        anomaly_maps_all = torch.mean(torch.cat(anomaly_maps_r_resized, dim=0), dim=0)  # [B, img_size, img_size]
        
        return anomaly_maps_all.numpy()
    
    def _run_rscin(self, anomaly_scores: np.ndarray, cls_tokens: torch.Tensor) -> np.ndarray:
        """
        执行 RsCIN 分数调整
        
        Args:
            anomaly_scores: [B] 原始异常分数
            cls_tokens: [B, dim] 类别令牌
            
        Returns:
            adjusted_scores: [B] 调整后的异常分数
        """
        musc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MuSc')
        if musc_dir not in sys.path:
            sys.path.insert(0, musc_dir)
        
        from MuSc.models.modules.RsCIN import RsCIN
        
        # 转换为 numpy
        cls_tokens_np = [cls_tokens[i].cpu().numpy() for i in range(len(cls_tokens))]
        
        # 使用 RsCIN 调整分数
        k_list = [1, 2, 3]  # MVTec AD 的默认配置
        adjusted_scores = RsCIN(anomaly_scores, cls_tokens_np, k_list=k_list)
        
        return adjusted_scores
    
    def predict(self, image_path: str) -> DetectionResult:
        """单张图像推理（使用参考集模拟完整流程）"""
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # 读取图像
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            img_resized = cv2.resize(img_np, (self.img_size, self.img_size))
            img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # 提取特征
            patch_tokens, image_features = self._extract_features(img_tensor)
            
            # 如果没有参考集，创建简单的伪参考集
            if self._reference_features is None:
                self._build_reference_set(patch_tokens, image_features)
            
            # 合并当前图像和参考集
            combined_patch_tokens = []
            for i, pt in enumerate(patch_tokens):
                ref_pt = self._reference_features[i].to(pt.device)
                combined = torch.cat([pt, ref_pt], dim=0)
                combined_patch_tokens.append(combined)
            
            combined_cls = torch.cat([image_features, self._reference_cls_tokens.to(image_features.device)], dim=0)
            
            # 执行完整 LNAMD + MSM 流程
            anomaly_maps = self._run_lnamd_msm(combined_patch_tokens)
            
            # 提取所有样本的异常分数（用于 RsCIN）
            ac_scores = np.array([am.reshape(-1).max() for am in anomaly_maps])
            
            # 当前图像的异常分数和热力图
            current_anomaly_map = anomaly_maps[0]  # [H, W]
            current_score = float(ac_scores[0])
            
            # 执行 RsCIN（需要至少2个样本）
            # RsCIN 需要所有样本的分数和 cls_tokens 数量一致
            if len(combined_cls) >= 2 and len(ac_scores) == len(combined_cls):
                try:
                    adjusted_scores = self._run_rscin(ac_scores, combined_cls)
                    final_score = float(adjusted_scores[0])
                except Exception as rscin_error:
                    print(f"[MuSc] Warning: RsCIN failed: {rscin_error}, using raw score")
                    final_score = current_score
            else:
                final_score = current_score
            
            # 判定是否异常
            is_anomaly = final_score > self.threshold
            
            inference_time = (time.time() - start_time) * 1000
            
            return DetectionResult(
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(final_score),
                anomaly_map=current_anomaly_map,
                inference_time=inference_time,
                metadata={
                    'backbone': self.backbone,
                    'image_size': self.img_size,
                    'method': 'musc-complete-lnamd-msm-rscin'
                }
            )
            
        except Exception as e:
            print(f"[ERROR] MuSc prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _build_reference_set(self, patch_tokens: List[torch.Tensor], cls_token: torch.Tensor, num_refs: int = 5):
        """
        构建伪参考集（单图推理时使用）- 使用简单的特征扰动
        
        注意：这不是原算法的做法，原算法使用完整的无标签测试集。
        仅在单图推理且没有真实参考集时使用。
        """
        ref_patch_tokens = []
        ref_cls_tokens = []
        
        torch.manual_seed(42)  # 保证可重复
        
        for _ in range(num_refs):
            perturbed_patches = []
            for pt in patch_tokens:
                # 添加小的随机扰动
                noise = torch.randn_like(pt) * 0.01
                perturbed_patches.append(pt + noise)
            ref_patch_tokens.append(perturbed_patches)
            
            # CLS token 扰动
            cls_noise = torch.randn_like(cls_token) * 0.01
            ref_cls_tokens.append(cls_token + cls_noise)
        
        # 合并参考集
        self._reference_features = []
        for i in range(len(patch_tokens)):
            layer_refs = torch.cat([ref_patch_tokens[j][i] for j in range(num_refs)], dim=0)
            self._reference_features.append(layer_refs)
        
        self._reference_cls_tokens = torch.cat(ref_cls_tokens, dim=0)
        print(f"[MuSc] Built pseudo reference set with {num_refs} perturbed samples")
    
    def predict_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """
        批量推理（使用真实样本作为参考集 - 更接近原算法逻辑）
        
        原算法假设有一个完整的无标签测试集，样本间互相作为参考。
        这里使用用户上传的所有样本作为互相参考集。
        """
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        if len(image_paths) < 2:
            # 图片数量不足，使用单张推理（带伪参考集）
            results = [self.predict(path) for path in image_paths]
            return results
        
        try:
            print(f"[MuSc] Processing {len(image_paths)} images as mutual reference set")
            
            # 准备图像
            images = []
            for path in image_paths:
                img = Image.open(path).convert('RGB')
                img_np = np.array(img)
                img_resized = cv2.resize(img_np, (self.img_size, self.img_size))
                img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1)
                images.append(img_tensor)
            
            images_batch = torch.stack(images).to(self.device)
            
            # 提取特征
            patch_tokens, image_features = self._extract_features(images_batch)
            
            # 使用所有样本作为参考集执行 LNAMD + MSM
            # 这与原算法逻辑一致：测试集内样本互相作为参考
            anomaly_maps = self._run_lnamd_msm(patch_tokens)  # [B, H, W]
            
            # 计算图像级异常分数（像素最大值）
            ac_scores = np.array([am.reshape(-1).max() for am in anomaly_maps])
            
            # 执行 RsCIN 调整
            adjusted_scores = self._run_rscin(ac_scores, image_features)
            
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(image_paths)
            
            results = []
            for i, (path, score, anomaly_map) in enumerate(zip(image_paths, adjusted_scores, anomaly_maps)):
                is_anomaly = score > self.threshold
                results.append(DetectionResult(
                    is_anomaly=bool(is_anomaly),
                    anomaly_score=float(score),
                    anomaly_map=anomaly_map,
                    inference_time=avg_time,
                    metadata={
                        'backbone': self.backbone,
                        'image_size': self.img_size,
                        'method': 'musc-mutual-reference',
                        'reference_count': len(image_paths)
                    }
                ))
            
            return results
            
        except Exception as e:
            print(f"[ERROR] MuSc batch prediction failed: {e}")
            import traceback
            traceback.print_exc()
            # 回退到单张推理
            results = [self.predict(path) for path in image_paths]
            return results
    
    def release(self) -> None:
        """释放资源"""
        self._musc = None
        self._reference_features = None
        self._reference_cls_tokens = None
        super().release()


# 注册不同backbone的MuSc变体
@register_algorithm("musc_clip_b32_512")
class MuScCLIP_B32_512(MuScBaseAdapter):
    """MuSc with CLIP ViT-B-32@512px - 小模型，更快"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='ViT-B-32', img_size=512, **kwargs)


@register_algorithm("musc_clip_b16_512")
class MuScCLIP_B16_512(MuScBaseAdapter):
    """MuSc with CLIP ViT-B-16@512px - 中等模型"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='ViT-B-16', img_size=512, **kwargs)


@register_algorithm("musc_clip_l14_336")
class MuScCLIP_L14_336(MuScBaseAdapter):
    """MuSc with CLIP ViT-L-14@336px - 推荐"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='ViT-L-14-336', img_size=336, **kwargs)


@register_algorithm("musc_clip_l14_518")
class MuScCLIP_L14_518(MuScBaseAdapter):
    """MuSc with CLIP ViT-L-14@518px - 更高精度"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='ViT-L-14-336', img_size=518, **kwargs)


@register_algorithm("musc_dinov2_b14_336")
class MuScDINOv2_B14_336(MuScBaseAdapter):
    """MuSc with DINOv2 ViT-B-14@336px - 轻量级"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2_vitb14', img_size=336, **kwargs)


@register_algorithm("musc_dinov2_b14_518")
class MuScDINOv2_B14_518(MuScBaseAdapter):
    """MuSc with DINOv2 ViT-B-14@518px - 中等精度"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2_vitb14', img_size=518, **kwargs)


@register_algorithm("musc_dinov2_l14_336")
class MuScDINOv2_L14_336(MuScBaseAdapter):
    """MuSc with DINOv2 ViT-L-14@336px - 高精度"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2_vitl14', img_size=336, **kwargs)


@register_algorithm("musc_dinov2_l14_518")
class MuScDINOv2_L14_518(MuScBaseAdapter):
    """MuSc with DINOv2 ViT-L-14@518px - 最高精度"""
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, backbone='dinov2_vitl14', img_size=518, **kwargs)


def create_musc_detector(algorithm_name: str, model_path: str, **kwargs):
    """创建MuSc检测器的工厂函数"""
    from algorithms import create_detector
    return create_detector(algorithm_name, model_path, **kwargs)
