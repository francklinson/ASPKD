#!/usr/bin/env python3
"""
预下载所有 Anomalib 模型的 backbone 权重到本地 HuggingFace 缓存

使用方法:
    python3 scripts/pre_download_backbones.py

缓存位置:
    ~/.cache/huggingface/hub/   (HuggingFace Hub 缓存)
    ~/.cache/open_clip/         (OpenCLIP 缓存)

下载后即可离线使用。
"""

import os
import sys
import time

# 设置路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algorithms'))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'pre_trained', 'huggingface'
)


def download_model(model_name: str) -> bool:
    """加载模型触发 backbone 下载"""
    from anomalib.models import get_model
    import torch

    print(f'  [{model_name}] Loading...', end=' ', flush=True)
    try:
        m = get_model(model_name)
        # 触发 backbone 初始化（部分模型需要前向传播才下载）
        m.to(torch.device('cuda'))
        print('✅')
        del m
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f'❌ {type(e).__name__}: {str(e)[:100]}')
        return False


def download_openclip():
    """下载 WinCLIP 需要的 OpenCLIP 权重"""
    try:
        import open_clip
        print('  [winclip/open_clip] Downloading ViT-B-16-plus-240...', end=' ', flush=True)
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-plus-240",
            pretrained="laion400m_e31",
        )
        print('✅')
    except Exception as e:
        print(f'❌ {type(e).__name__}: {str(e)[:100]}')


def download_anomalyvfm():
    """下载 AnomalyVFM 的 RADIO 权重 (HuggingFace)"""
    try:
        from huggingface_hub import hf_hub_download
        print('  [anomalyvfm] Downloading RADIO model.safetensors...', end=' ', flush=True)
        path = hf_hub_download(
            repo_id="MaticFuc/anomalyvfm_radio",
            filename="model.safetensors",
            revision="17654e763c8fae5ae1c44e2ec421a427783d6196",
        )
        print(f'✅ -> {path}')
    except Exception as e:
        print(f'❌ {type(e).__name__}: {str(e)[:100]}')


def main():
    print('=' * 60)
    print('Anomalib Backbone 预下载')
    print(f'缓存目录: {os.environ["HF_HOME"]}')
    print('=' * 60)

    # ---- 需要 timm backbone 的模型 (从 HF Hub 下载) ----
    timm_models = [
        # 模型名           # backbone
        'dfm',             # resnet50
        'fre',             # resnet50
        'supersimplenet',  # wide_resnet50_2
        'patchflow',       # efficientnet_b5
        'general_ad',      # vit_large_patch14_dinov2
        'cfm',             # vit_base_patch8_224.dino + Point-MAE
        'cfa',             # resnet18/wide_resnet50_2
        'patchcore',       # wide_resnet50_2
        'padim',           # resnet18
        'dfkde',           # resnet18
        'uflow',           # cait_m48_448 + cait_s24_224
    ]

    print('\n[1/3] 下载 timm backbone 模型 (HF Hub)...')
    success = 0
    for name in timm_models:
        if download_model(name):
            success += 1
    print(f'  完成: {success}/{len(timm_models)}')

    # ---- OpenCLIP (WinCLIP) ----
    print('\n[2/3] 下载 OpenCLIP 权重...')
    download_openclip()

    # ---- AnomalyVFM (HF Hub) ----
    print('\n[3/3] 下载 AnomalyVFM 权重...')
    download_anomalyvfm()

    print('\n' + '=' * 60)
    print('预下载完成！')
    print(f'HF 缓存: {os.environ["HF_HOME"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
