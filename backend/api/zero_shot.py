"""
Zero-Shot 异常检测 API
基于 MuSc (ICLR 2024) 实现零样本工业异常检测
"""
import os
import sys
import time
import shutil
import tempfile
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid

router = APIRouter()

# 支持的backbone类型
BACKBONES = {
    # CLIP ViT-B系列 (小模型，更快)
    "musc_clip_b32_512": "MuSc + CLIP ViT-B/32@512px (快速)",
    "musc_clip_b16_512": "MuSc + CLIP ViT-B/16@512px (均衡)",
    # CLIP ViT-L系列 (大模型，高精度)
    "musc_clip_l14_336": "MuSc + CLIP ViT-L/14@336px (推荐)",
    "musc_clip_l14_518": "MuSc + CLIP ViT-L/14@518px (高精度)",
    # DINOv2 ViT-B系列 (Base模型)
    "musc_dinov2_b14_336": "MuSc + DINOv2 ViT-B/14@336px (轻量)",
    "musc_dinov2_b14_518": "MuSc + DINOv2 ViT-B/14@518px (均衡)",
    # DINOv2 ViT-L系列 (Large模型)
    "musc_dinov2_l14_336": "MuSc + DINOv2 ViT-L/14@336px (高精度)",
    "musc_dinov2_l14_518": "MuSc + DINOv2 ViT-L/14@518px (最高精度)",
}


class ZeroShotRequest(BaseModel):
    """零样本检测请求"""
    backbone: str = "musc_clip_l14_336"
    threshold: float = 0.5
    batch_size: int = 4
    r_list: str = "1,3,5"  # 聚合度列表


class ZeroShotResponse(BaseModel):
    """零样本检测响应"""
    task_id: str
    status: str
    message: str


class ZeroShotResult(BaseModel):
    """零样本检测结果"""
    task_id: str
    status: str
    progress: float
    results: Optional[List[Dict[str, Any]]] = None
    report: Optional[str] = None
    error: Optional[str] = None


def log_operation(operation: str, details: str = "", status: str = "INFO"):
    """记录操作日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [ZeroShot] [{status}] {operation} | {details}")


@router.get("/backbones")
async def get_available_backbones():
    """获取可用的backbone列表"""
    return {
        "backbones": [
            {"id": key, "name": value}
            for key, value in BACKBONES.items()
        ]
    }


@router.post("/analyze", response_model=ZeroShotResponse)
async def analyze_zero_shot(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    backbone: str = Form("musc_clip_l14_336"),
    threshold: float = Form(0.5),
    batch_size: int = Form(4),
    r_list: str = Form("1,3,5")
):
    """
    上传图像并进行零样本异常检测

    - **files**: 图像文件列表 (支持 png, jpg, jpeg, bmp 等格式)
    - **backbone**: 骨干网络类型
    - **threshold**: 异常判定阈值
    - **batch_size**: 批处理大小
    - **r_list**: 聚合度列表 (逗号分隔)
    """
    start_time = time.time()
    task_id = str(uuid.uuid4())

    log_operation(
        "ANALYZE_START",
        f"任务ID={task_id}, backbone={backbone}, 文件数={len(files)}"
    )

    # 验证backbone类型
    if backbone not in BACKBONES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的backbone类型: {backbone}"
        )

    # 验证文件
    if not files or len(files) < 1:
        raise HTTPException(
            status_code=400,
            detail="请至少上传1个图像文件"
        )

    # MuSc 算法需要至少5个样本作为参考集
    if len(files) < 5:
        raise HTTPException(
            status_code=400,
            detail="MuSc 零样本检测需要至少 5 个图像文件才能进行有效分析。当前文件数: {}".format(len(files))
        )

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
    valid_files = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in allowed_extensions:
            valid_files.append(file)

    if not valid_files:
        raise HTTPException(
            status_code=400,
            detail="请上传有效的图像文件"
        )

    # 解析r_list参数
    try:
        r_list = [int(x.strip()) for x in r_list.split(',') if x.strip()]
    except ValueError:
        r_list = [1, 3, 5]

    # 创建任务目录
    task_dir = os.path.join("uploads", "zero_shot", task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 保存上传的文件
    saved_files = []
    try:
        for file in valid_files:
            file_path = os.path.join(task_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_files.append(file_path)

        log_operation(
            "FILES_SAVED",
            f"任务ID={task_id}, 保存了 {len(saved_files)} 个文件"
        )

        # 启动后台分析任务
        background_tasks.add_task(
            run_zero_shot_analysis,
            task_id=task_id,
            file_paths=saved_files,
            backbone=backbone,
            threshold=threshold,
            batch_size=batch_size,
            r_list=r_list
        )

        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "ANALYZE_QUEUED",
            f"任务ID={task_id}, 耗时={elapsed_time:.2f}ms"
        )

        return ZeroShotResponse(
            task_id=task_id,
            status="queued",
            message=f"零样本分析任务已创建，共 {len(saved_files)} 个文件"
        )

    except Exception as e:
        # 清理临时文件
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
        log_operation(
            "ANALYZE_ERROR",
            f"任务ID={task_id}, 错误={str(e)}",
            "ERROR"
        )
        raise HTTPException(status_code=500, detail=f"创建分析任务失败: {str(e)}")


def run_zero_shot_analysis(
    task_id: str,
    file_paths: List[str],
    backbone: str,
    threshold: float,
    batch_size: int,
    r_list: List[int]
):
    """运行零样本分析（后台任务）"""
    import json
    import cv2
    
    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from algorithms import create_detector
    
    start_time = time.time()
    log_operation("ANALYSIS_START", f"任务ID={task_id}, 开始零样本分析")
    log_operation("PARAMS", f"任务ID={task_id}, backbone={backbone}, threshold={threshold}, "
                    f"batch_size={batch_size}, r_list={r_list}")

    try:
        # 创建检测器
        log_operation("DETECTOR_INIT", f"任务ID={task_id}, 正在初始化检测器: {backbone}...")
        
        # 使用临时模型路径（MuSc是预训练的，不需要本地模型文件）
        # 创建一个虚拟路径用于初始化
        temp_model_path = os.path.join("uploads", "zero_shot", task_id, "model_placeholder.pth")
        
        detector = create_detector(
            backbone,
            model_path=temp_model_path,
            batch_size=batch_size,
            r_list=r_list
        )
        
        # 覆盖默认阈值
        detector.threshold = threshold
        
        log_operation("DETECTOR_INIT", f"任务ID={task_id}, 检测器初始化完成")

        # 加载模型
        log_operation("MODEL_LOAD", f"任务ID={task_id}, 正在加载模型...")
        detector.load_model()
        log_operation("MODEL_LOAD", f"任务ID={task_id}, 模型加载完成")

        # 创建结果目录（使用绝对路径）
        result_dir = os.path.abspath(os.path.join("visualize", "zero_shot", task_id))
        os.makedirs(result_dir, exist_ok=True)
        log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, 结果目录: {result_dir}")
        
        # 执行推理
        log_operation("INFERENCE", f"任务ID={task_id}, 开始推理, 共 {len(file_paths)} 个文件")
        
        results = []
        
        # 使用批量推理，让所有样本互相作为参考
        log_operation("BATCH_INFERENCE", f"任务ID={task_id}, 使用批量推理（样本间互相参考）")
        batch_results = detector.predict_batch(file_paths)
        
        for i, (file_path, result) in enumerate(zip(file_paths, batch_results)):
            try:
                log_operation("FILE_PROCESSING", f"任务ID={task_id}, [{i+1}/{len(file_paths)}] 处理中: {os.path.basename(file_path)}")
                
                # 生成热力图
                heatmap_url = None
                overlay_url = None
                original_url = None
                log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, anomaly_map 类型: {type(result.anomaly_map)}, 是否为None: {result.anomaly_map is None}")
                
                # 先读取原图（独立于热力图生成）
                try:
                    import cv2
                    original_img_bgr = cv2.imread(file_path)
                    if original_img_bgr is not None:
                        # 保存原图
                        original_filename = f"original_{i:03d}.png"
                        original_path = os.path.abspath(os.path.join(result_dir, original_filename))
                        os.makedirs(os.path.dirname(original_path), exist_ok=True)
                        cv2.imwrite(original_path, original_img_bgr)
                        if os.path.exists(original_path):
                            original_url = f"/visualize/zero_shot/{task_id}/{original_filename}"
                            log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, 原图已保存: {original_url}")
                except Exception as original_error:
                    log_operation("HEATMAP_WARN", f"任务ID={task_id}, 原图保存失败: {str(original_error)}", "WARNING")
                
                try:
                    if result.anomaly_map is not None:
                        import matplotlib.pyplot as plt
                        from matplotlib import cm
                        
                        log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, anomaly_map shape: {result.anomaly_map.shape}, dtype: {result.anomaly_map.dtype}")
                        
                        # 确保原图已读取
                        if 'original_img_bgr' not in locals() or original_img_bgr is None:
                            original_img_bgr = cv2.imread(file_path)
                            if original_img_bgr is None:
                                raise ValueError(f"无法读取图像: {file_path}")
                        original_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
                        h, w = original_img.shape[:2]
                        
                        # 调整热力图尺寸到原图大小
                        anomaly_map_resized = cv2.resize(result.anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # 归一化到 0-1 范围（遵循原算法 logic）
                        amap_min = anomaly_map_resized.min()
                        amap_max = anomaly_map_resized.max()
                        if amap_max > amap_min:
                            anomaly_map_norm = (anomaly_map_resized - amap_min) / (amap_max - amap_min)
                        else:
                            anomaly_map_norm = np.zeros_like(anomaly_map_resized)
                        
                        log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, anomaly_map 原始范围: [{amap_min:.4f}, {amap_max:.4f}]")
                        
                        # 1. 生成独立的热力图
                        heatmap_filename = f"heatmap_{i:03d}.png"
                        heatmap_path = os.path.abspath(os.path.join(result_dir, heatmap_filename))
                        
                        log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, 保存热力图到: {heatmap_path}")
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                        
                        # 使用归一化后的热力图显示
                        plt.figure(figsize=(6, 6))
                        plt.imshow(anomaly_map_norm, cmap='jet', vmin=0, vmax=1)
                        plt.colorbar(label='Anomaly Score')
                        plt.title(f"Anomaly Map\nScore: {result.anomaly_score:.4f}")
                        plt.axis('off')
                        plt.savefig(heatmap_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                        
                        if os.path.exists(heatmap_path):
                            heatmap_url = f"/visualize/zero_shot/{task_id}/{heatmap_filename}"
                            log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, 热力图已保存, URL: {heatmap_url}")
                        else:
                            log_operation("HEATMAP_ERROR", f"任务ID={task_id}, 热力图保存失败，文件不存在", "ERROR")
                        
                        # 2. 生成叠加热力图
                        overlay_filename = f"overlay_{i:03d}.png"
                        overlay_path = os.path.abspath(os.path.join(result_dir, overlay_filename))
                        
                        # 将归一化后的热力图转换为彩色 (jet colormap)
                        heatmap_color = cm.jet(anomaly_map_norm)[:, :, :3]  # RGB, 去掉alpha通道
                        heatmap_color = (heatmap_color * 255).astype(np.uint8)
                        
                        # 创建叠加图像 (alpha混合)
                        alpha = 0.6  # 热力图透明度
                        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
                        
                        # 保存叠加图像
                        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(overlay_path, overlay_bgr)
                        
                        if os.path.exists(overlay_path):
                            overlay_url = f"/visualize/zero_shot/{task_id}/{overlay_filename}"
                            log_operation("HEATMAP_DEBUG", f"任务ID={task_id}, 叠加图已保存, URL: {overlay_url}")
                        
                        log_operation("HEATMAP_SAVED", f"任务ID={task_id}, 热力图已保存: {heatmap_filename}, 叠加图: {overlay_filename}")
                    else:
                        log_operation("HEATMAP_WARN", f"任务ID={task_id}, anomaly_map 为 None，跳过热力图生成", "WARNING")
                except Exception as heatmap_error:
                    log_operation("HEATMAP_ERROR", f"任务ID={task_id}, 热力图生成失败: {str(heatmap_error)}", "WARNING")
                    import traceback
                    log_operation("HEATMAP_ERROR_DETAIL", f"任务ID={task_id}, 错误详情: {traceback.format_exc()}", "ERROR")
                
                results.append({
                    "filename": os.path.basename(file_path),
                    "is_anomaly": result.is_anomaly,
                    "anomaly_score": result.anomaly_score,
                    "inference_time": result.inference_time,
                    "heatmap_url": heatmap_url,
                    "overlay_url": overlay_url,
                    "original_url": original_url,
                    "metadata": result.metadata
                })
                
                log_operation("FILE_SUCCESS", f"任务ID={task_id}, [{i+1}/{len(file_paths)}] 处理成功: "
                                              f"score={result.anomaly_score:.4f}, "
                                              f"anomaly={result.is_anomaly}, "
                                              f"time={result.inference_time:.2f}ms")
            except Exception as e:
                log_operation("FILE_ERROR", f"任务ID={task_id}, [{i+1}/{len(file_paths)}] 文件处理失败: "
                                          f"{os.path.basename(file_path)}, 错误: {str(e)}", "ERROR")
                import traceback
                log_operation("FILE_ERROR_DETAIL", f"任务ID={task_id}, 错误详情: {traceback.format_exc()}", "ERROR")
                results.append({
                    "filename": os.path.basename(file_path),
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "inference_time": 0.0,
                    "heatmap_url": None,
                    "error": str(e)
                })

        log_operation("INFERENCE", f"任务ID={task_id}, 推理完成")

        # 计算统计信息
        anomaly_count = sum(1 for r in results if r.get('is_anomaly', False))
        avg_score = np.mean([r['anomaly_score'] for r in results])
        avg_time = np.mean([r['inference_time'] for r in results])

        # 生成报告
        report = generate_report(file_paths, results, backbone, threshold)

        # 保存结果
        result_data = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "results": results,
            "summary": {
                "total_files": len(file_paths),
                "anomaly_count": anomaly_count,
                "normal_count": len(file_paths) - anomaly_count,
                "avg_anomaly_score": float(avg_score),
                "avg_inference_time": float(avg_time),
                "backbone": backbone,
                "threshold": threshold
            },
            "report": report
        }

        # 保存结果到文件
        result_file = os.path.join(result_dir, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        elapsed_time = (time.time() - start_time)
        log_operation(
            "ANALYSIS_SUCCESS",
            f"任务ID={task_id}, 耗时={elapsed_time:.2f}s, 异常数={anomaly_count}"
        )

        # 释放检测器资源
        detector.release()

    except Exception as e:
        log_operation(
            "ANALYSIS_ERROR",
            f"任务ID={task_id}, 错误={str(e)}",
            "ERROR"
        )
        import traceback
        log_operation("ANALYSIS_ERROR_DETAIL", f"任务ID={task_id}, 错误详情: {traceback.format_exc()}", "ERROR")
        
        # 保存错误信息
        result_data = {
            "task_id": task_id,
            "status": "failed",
            "progress": 0,
            "error": str(e)
        }
        result_dir = os.path.join("visualize", "zero_shot", task_id)
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(result_dir, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)


def generate_report(file_paths, results, backbone, threshold):
    """生成分析报告"""
    report_lines = [f"# 零样本异常检测报告 ({backbone})", ""]
    report_lines.append(f"## 检测概况")
    report_lines.append(f"- 检测文件数: {len(file_paths)}")
    report_lines.append(f"- 检测异常数: {sum(1 for r in results if r.get('is_anomaly', False))}")
    report_lines.append(f"- 正常数: {sum(1 for r in results if not r.get('is_anomaly', False))}")
    report_lines.append(f"- 阈值: {threshold}")
    report_lines.append("")

    report_lines.append(f"## 详细结果")
    report_lines.append("| 文件名 | 异常分数 | 是否异常 | 推理时间(ms) |")
    report_lines.append("|--------|----------|----------|---------------|")
    
    for path, result in zip(file_paths, results):
        filename = os.path.basename(path)
        score = result.get('anomaly_score', 0.0)
        is_anomaly = result.get('is_anomaly', False)
        inference_time = result.get('inference_time', 0.0)
        anomaly_str = "是" if is_anomaly else "否"
        report_lines.append(f"| {filename} | {score:.4f} | {anomaly_str} | {inference_time:.2f} |")

    return "\n".join(report_lines)


@router.get("/result/{task_id}")
async def get_zero_shot_result(task_id: str):
    """获取零样本分析结果"""
    result_dir = os.path.join("visualize", "zero_shot", task_id)
    result_file = os.path.join(result_dir, "result.json")

    if not os.path.exists(result_file):
        # 检查任务是否还在进行中
        task_dir = os.path.join("uploads", "zero_shot", task_id)
        if os.path.exists(task_dir):
            return {
                "task_id": task_id,
                "status": "processing",
                "progress": 50.0,
                "message": "分析正在进行中..."
            }
        else:
            raise HTTPException(status_code=404, detail="任务不存在")

    import json
    with open(result_file, "r", encoding="utf-8") as f:
        result = json.load(f)

    return result
