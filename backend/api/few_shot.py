"""
Few-Shot (少样本) 异常检测 API
基于 SubspaceAD (CVPR 2026) 实现少样本工业异常检测
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

# 支持的 backbone 类型
BACKBONES = {
    # DINOv2 Large 系列 (高精度)
    "subspacead_dinov2_large_672": "SubspaceAD + DINOv2-Large@672px (推荐)",
    "subspacead_dinov2_large_518": "SubspaceAD + DINOv2-Large@518px (高精度)",
    "subspacead_dinov2_large_336": "SubspaceAD + DINOv2-Large@336px (均衡)",
    # DINOv2 Base 系列 (轻量级)
    "subspacead_dinov2_base_672": "SubspaceAD + DINOv2-Base@672px (轻量)",
    "subspacead_dinov2_base_518": "SubspaceAD + DINOv2-Base@518px (轻量均衡)",
    # DINOv2 Small 系列 (快速)
    "subspacead_dinov2_small_672": "SubspaceAD + DINOv2-Small@672px (快速)",
}


class FewShotRequest(BaseModel):
    """少样本检测请求"""
    backbone: str = "subspacead_dinov2_large_672"
    threshold: float = 0.5
    k_shot: int = 1
    pca_ev: float = 0.99
    score_method: str = "reconstruction"


class FewShotResponse(BaseModel):
    """少样本检测响应"""
    task_id: str
    status: str
    message: str


class FewShotResult(BaseModel):
    """少样本检测结果"""
    task_id: str
    status: str
    progress: float
    results: Optional[List[Dict[str, Any]]] = None
    report: Optional[str] = None
    error: Optional[str] = None


def log_operation(operation: str, details: str = "", status: str = "INFO"):
    """记录操作日志"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [FewShot] [{status}] {operation} | {details}")


@router.get("/backbones")
async def get_available_backbones():
    """获取可用的 backbone 列表"""
    return {
        "backbones": [
            {"id": key, "name": value}
            for key, value in BACKBONES.items()
        ]
    }


@router.post("/analyze", response_model=FewShotResponse)
async def analyze_few_shot(
    background_tasks: BackgroundTasks,
    reference_files: List[UploadFile] = File(...),
    test_files: List[UploadFile] = File(...),
    backbone: str = Form("subspacead_dinov2_large_672"),
    threshold: float = Form(0.5),
    k_shot: int = Form(1),
    pca_ev: float = Form(0.99),
    score_method: str = Form("reconstruction")
):
    """
    上传图像进行少样本异常检测

    - **reference_files**: 参考图像文件列表（正常样本，至少1个）
    - **test_files**: 测试图像文件列表（待检测）
    - **backbone**: 骨干网络类型
    - **threshold**: 异常判定阈值
    - **k_shot**: 使用的参考样本数量
    - **pca_ev**: PCA 解释方差比例
    - **score_method**: 异常评分方法 (reconstruction/mahalanobis/euclidean/cosine)
    """
    start_time = time.time()
    task_id = str(uuid.uuid4())

    log_operation(
        "ANALYZE_START",
        f"任务ID={task_id}, backbone={backbone}, 参考文件数={len(reference_files)}, 测试文件数={len(test_files)}"
    )

    # 验证 backbone 类型
    if backbone not in BACKBONES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 backbone 类型: {backbone}"
        )

    # 验证参考文件
    if not reference_files or len(reference_files) < 1:
        raise HTTPException(
            status_code=400,
            detail="请至少上传 1 个参考图像文件（正常样本）"
        )

    # 验证测试文件
    if not test_files or len(test_files) < 1:
        raise HTTPException(
            status_code=400,
            detail="请至少上传 1 个测试图像文件"
        )

    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
    
    # 验证参考文件
    valid_ref_files = []
    for file in reference_files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in allowed_extensions:
            valid_ref_files.append(file)
    
    if not valid_ref_files:
        raise HTTPException(
            status_code=400,
            detail="请上传有效的参考图像文件"
        )

    # 验证测试文件
    valid_test_files = []
    for file in test_files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in allowed_extensions:
            valid_test_files.append(file)
    
    if not valid_test_files:
        raise HTTPException(
            status_code=400,
            detail="请上传有效的测试图像文件"
        )

    # 创建任务目录
    task_dir = os.path.join("uploads", "few_shot", task_id)
    os.makedirs(task_dir, exist_ok=True)

    # 保存上传的文件
    saved_ref_files = []
    saved_test_files = []
    
    try:
        # 保存参考文件
        ref_dir = os.path.join(task_dir, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        for file in valid_ref_files:
            file_path = os.path.join(ref_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_ref_files.append(file_path)

        # 保存测试文件
        test_dir = os.path.join(task_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        for file in valid_test_files:
            file_path = os.path.join(test_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_test_files.append(file_path)

        log_operation(
            "FILES_SAVED",
            f"任务ID={task_id}, 保存了 {len(saved_ref_files)} 个参考文件, {len(saved_test_files)} 个测试文件"
        )

        # 启动后台分析任务
        background_tasks.add_task(
            run_few_shot_analysis,
            task_id=task_id,
            ref_paths=saved_ref_files,
            test_paths=saved_test_files,
            backbone=backbone,
            threshold=threshold,
            k_shot=k_shot,
            pca_ev=pca_ev,
            score_method=score_method
        )

        elapsed_time = (time.time() - start_time) * 1000
        log_operation(
            "ANALYZE_QUEUED",
            f"任务ID={task_id}, 耗时={elapsed_time:.2f}ms"
        )

        return FewShotResponse(
            task_id=task_id,
            status="queued",
            message=f"少样本分析任务已创建，参考样本: {len(saved_ref_files)}，测试样本: {len(saved_test_files)}"
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


def run_few_shot_analysis(
    task_id: str,
    ref_paths: List[str],
    test_paths: List[str],
    backbone: str,
    threshold: float,
    k_shot: int,
    pca_ev: float,
    score_method: str
):
    """运行少样本分析（后台任务）"""
    import json
    import cv2
    
    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from algorithms import create_detector
    
    start_time = time.time()
    log_operation("ANALYSIS_START", f"任务ID={task_id}, 开始少样本分析")
    log_operation("PARAMS", f"任务ID={task_id}, backbone={backbone}, threshold={threshold}, "
                    f"k_shot={k_shot}, pca_ev={pca_ev}, score_method={score_method}")

    try:
        # 创建检测器
        log_operation("DETECTOR_INIT", f"任务ID={task_id}, 正在初始化检测器: {backbone}...")
        
        # 创建虚拟模型路径
        temp_model_path = os.path.join("uploads", "few_shot", task_id, "model_placeholder.pth")
        
        detector = create_detector(
            backbone,
            model_path=temp_model_path,
            threshold=threshold,
            k_shot=k_shot,
            pca_ev=pca_ev,
            score_method=score_method
        )
        
        log_operation("DETECTOR_INIT", f"任务ID={task_id}, 检测器初始化完成")

        # 加载模型
        log_operation("MODEL_LOAD", f"任务ID={task_id}, 正在加载模型...")
        detector.load_model()
        log_operation("MODEL_LOAD", f"任务ID={task_id}, 模型加载完成")

        # 创建结果目录
        result_dir = os.path.abspath(os.path.join("visualize", "few_shot", task_id))
        os.makedirs(result_dir, exist_ok=True)
        log_operation("RESULT_DIR", f"任务ID={task_id}, 结果目录: {result_dir}")
        
        # 执行批量推理
        log_operation("INFERENCE", f"任务ID={task_id}, 开始推理, {len(test_paths)} 个测试文件")
        
        # 限制参考样本数量
        if len(ref_paths) > k_shot:
            ref_paths = ref_paths[:k_shot]
        
        results = detector.predict_batch(test_paths, ref_paths)
        
        # 处理结果
        processed_results = []
        for i, (file_path, result) in enumerate(zip(test_paths, results)):
            try:
                log_operation("FILE_PROCESSING", f"任务ID={task_id}, [{i+1}/{len(test_paths)}] 处理中: {os.path.basename(file_path)}")
                
                # 生成热力图
                heatmap_url = None
                overlay_url = None
                original_url = None
                
                # 读取原图
                try:
                    original_img_bgr = cv2.imread(file_path)
                    if original_img_bgr is not None:
                        original_filename = f"original_{i:03d}.png"
                        original_path = os.path.abspath(os.path.join(result_dir, original_filename))
                        cv2.imwrite(original_path, original_img_bgr)
                        if os.path.exists(original_path):
                            original_url = f"/visualize/few_shot/{task_id}/{original_filename}"
                except Exception as original_error:
                    log_operation("HEATMAP_WARN", f"任务ID={task_id}, 原图保存失败: {str(original_error)}", "WARNING")
                
                # 生成热力图
                try:
                    if result.anomaly_map is not None:
                        import matplotlib.pyplot as plt
                        from matplotlib import cm
                        
                        # 读取原图
                        if 'original_img_bgr' not in locals() or original_img_bgr is None:
                            original_img_bgr = cv2.imread(file_path)
                        original_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
                        h, w = original_img.shape[:2]
                        
                        # 调整热力图尺寸
                        anomaly_map_resized = cv2.resize(result.anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # 归一化
                        amap_min = anomaly_map_resized.min()
                        amap_max = anomaly_map_resized.max()
                        if amap_max > amap_min:
                            anomaly_map_norm = (anomaly_map_resized - amap_min) / (amap_max - amap_min)
                        else:
                            anomaly_map_norm = np.zeros_like(anomaly_map_resized)
                        
                        # 生成独立热力图
                        heatmap_filename = f"heatmap_{i:03d}.png"
                        heatmap_path = os.path.abspath(os.path.join(result_dir, heatmap_filename))
                        
                        plt.figure(figsize=(6, 6))
                        plt.imshow(anomaly_map_norm, cmap='jet', vmin=0, vmax=1)
                        plt.colorbar(label='Anomaly Score')
                        plt.title(f"Anomaly Map\nScore: {result.anomaly_score:.4f}")
                        plt.axis('off')
                        plt.savefig(heatmap_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
                        plt.close()
                        
                        if os.path.exists(heatmap_path):
                            heatmap_url = f"/visualize/few_shot/{task_id}/{heatmap_filename}"
                        
                        # 生成叠加热力图
                        overlay_filename = f"overlay_{i:03d}.png"
                        overlay_path = os.path.abspath(os.path.join(result_dir, overlay_filename))
                        
                        heatmap_color = cm.jet(anomaly_map_norm)[:, :, :3]
                        heatmap_color = (heatmap_color * 255).astype(np.uint8)
                        
                        alpha = 0.6
                        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
                        
                        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(overlay_path, overlay_bgr)
                        
                        if os.path.exists(overlay_path):
                            overlay_url = f"/visualize/few_shot/{task_id}/{overlay_filename}"
                        
                        log_operation("HEATMAP_SAVED", f"任务ID={task_id}, 热力图已保存: {heatmap_filename}")
                    else:
                        log_operation("HEATMAP_WARN", f"任务ID={task_id}, anomaly_map 为 None", "WARNING")
                except Exception as heatmap_error:
                    log_operation("HEATMAP_ERROR", f"任务ID={task_id}, 热力图生成失败: {str(heatmap_error)}", "WARNING")
                
                processed_results.append({
                    "filename": os.path.basename(file_path),
                    "is_anomaly": result.is_anomaly,
                    "anomaly_score": result.anomaly_score,
                    "inference_time": result.inference_time,
                    "heatmap_url": heatmap_url,
                    "overlay_url": overlay_url,
                    "original_url": original_url,
                    "metadata": result.metadata
                })
                
                log_operation("FILE_SUCCESS", f"任务ID={task_id}, [{i+1}/{len(test_paths)}] 处理成功: "
                                              f"score={result.anomaly_score:.4f}, anomaly={result.is_anomaly}")
            except Exception as e:
                log_operation("FILE_ERROR", f"任务ID={task_id}, [{i+1}/{len(test_paths)}] 文件处理失败: "
                                          f"{os.path.basename(file_path)}, 错误: {str(e)}", "ERROR")
                processed_results.append({
                    "filename": os.path.basename(file_path),
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "inference_time": 0.0,
                    "heatmap_url": None,
                    "overlay_url": None,
                    "error": str(e)
                })

        log_operation("INFERENCE", f"任务ID={task_id}, 推理完成")

        # 计算统计信息
        anomaly_count = sum(1 for r in processed_results if r.get('is_anomaly', False))
        avg_score = np.mean([r['anomaly_score'] for r in processed_results])
        avg_time = np.mean([r['inference_time'] for r in processed_results])

        # 生成报告
        report = generate_report(test_paths, processed_results, backbone, threshold, k_shot)

        # 保存结果
        result_data = {
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "results": processed_results,
            "summary": {
                "total_files": len(test_paths),
                "anomaly_count": anomaly_count,
                "normal_count": len(test_paths) - anomaly_count,
                "avg_anomaly_score": float(avg_score),
                "avg_inference_time": float(avg_time),
                "backbone": backbone,
                "threshold": threshold,
                "k_shot": k_shot,
                "pca_ev": pca_ev
            },
            "report": report
        }

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
        result_dir = os.path.join("visualize", "few_shot", task_id)
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(result_dir, "result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)


def generate_report(file_paths, results, backbone, threshold, k_shot):
    """生成分析报告"""
    report_lines = [f"# 少样本异常检测报告 ({backbone})", ""]
    report_lines.append(f"## 检测概况")
    report_lines.append(f"- 检测文件数: {len(file_paths)}")
    report_lines.append(f"- 参考样本数 (k-shot): {k_shot}")
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
async def get_few_shot_result(task_id: str):
    """获取少样本分析结果"""
    result_dir = os.path.join("visualize", "few_shot", task_id)
    result_file = os.path.join(result_dir, "result.json")

    if not os.path.exists(result_file):
        # 检查任务是否还在进行中
        task_dir = os.path.join("uploads", "few_shot", task_id)
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
