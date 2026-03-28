#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一算法调用示例 - 展示如何使用统一的接口调用不同的ASD算法

本示例演示了：
1. 创建检测器实例
2. 加载模型
3. 单张图像推理
4. 批量图像推理
5. 获取算法列表和信息
6. 处理检测结果
"""

import os
import sys
from typing import List, Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from algorithms import create_detector, list_available_algorithms, get_algorithm_info
from core.base_detector import DetectionResult


def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("示例1: 基础使用 - 单张图像检测")
    print("=" * 60)
    
    try:
        # 1. 创建检测器 (使用Dinomaly DINOv3 Small作为示例)
        print("\n[1/3] 创建检测器实例...")
        detector = create_detector("dinomaly_dinov3_small")
        print(f"✓ 检测器创建成功: {detector.__class__.__name__}")
        
        # 2. 加载模型
        print("\n[2/3] 加载模型...")
        detector.load_model()
        print("✓ 模型加载成功")
        
        # 3. 准备测试图像
        test_image = "test_data/sample_image.png"  # 请替换为实际图像路径
        if not os.path.exists(test_image):
            print(f"⚠  测试图像不存在: {test_image}")
            print("   请修改test_image路径为实际图像文件")
            return
        
        # 4. 执行推理
        print(f"\n[3/3] 执行推理: {test_image}")
        result = detector.predict(test_image)
        
        # 5. 显示结果
        print("\n" + "=" * 40)
        print("检测结果:")
        print("=" * 40)
        print(f"是否异常: {'是' if result.is_anomaly else '否'}")
        print(f"异常分数: {result.anomaly_score:.4f}")
        print(f"判定阈值: {detector.threshold:.4f}")
        print(f"推理时间: {result.inference_time:.2f} ms")
        print(f"运行设备: {detector.device}")
        
        # 6. 释放资源
        detector.release()
        print("\n✓ 资源已释放")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_batch_inference():
    """批量推理示例"""
    print("\n\n" + "=" * 60)
    print("示例2: 批量推理")
    print("=" * 60)
    
    try:
        # 1. 创建检测器 (使用ADer的MambaAD作为示例)
        print("\n[1/3] 创建检测器实例...")
        detector = create_detector("mambaad")
        print(f"✓ 检测器创建成功: {detector.__class__.__name__}")
        
        # 2. 加载模型
        print("\n[2/3] 加载模型...")
        detector.load_model()
        print("✓ 模型加载成功")
        
        # 3. 准备测试图像列表
        test_images = [
            "test_data/image1.png",
            "test_data/image2.png",
            "test_data/image3.png"
        ]
        
        # 过滤存在的文件
        valid_images = [img for img in test_images if os.path.exists(img)]
        
        if not valid_images:
            print("⚠  没有找到有效的测试图像")
            print("   请在test_data目录下放置测试图像，或修改test_images列表")
            return
        
        print(f"\n[3/3] 批量推理 ({len(valid_images)} 张图像)...")
        results = detector.predict_batch(valid_images)
        
        # 4. 显示结果
        print("\n" + "=" * 60)
        print("批量检测结果:")
        print("=" * 60)
        for idx, (image_path, result) in enumerate(zip(valid_images, results)):
            filename = os.path.basename(image_path)
            status = "🔴 异常" if result.is_anomaly else "🟢 正常"
            print(f"{idx+1}. {filename:<30} {status} | 分数: {result.anomaly_score:.4f}")
        
        # 5. 释放资源
        detector.release()
        print("\n✓ 资源已释放")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_algorithm_list():
    """获取算法列表示例"""
    print("\n\n" + "=" * 60)
    print("示例3: 获取可用算法列表")
    print("=" * 60)
    
    # 获取所有可用算法
    algorithms = list_available_algorithms()
    
    print(f"\n可用算法总数: {len(algorithms)}")
    print("\n" + "-" * 60)
    
    # 按类别分组显示
    categories = {
        "Dinomaly": [algo for algo in algorithms if algo.startswith("dinomaly")],
        "ADer": [algo for algo in algorithms if algo in ["mambaad", "invad", "vitad", "unad", "cflow", "pyramidflow", "simplenet"]],
        "Anomalib": [algo for algo in algorithms if algo in ["patchcore", "efficient_ad", "padim"]],
        "BaseASD": [algo for algo in algorithms if algo in ["denseae", "cae", "vae"]]
    }
    
    for category, algos in categories.items():
        if algos:
            print(f"\n{category} 系列:")
            for algo in algos:
                info = get_algorithm_info(algo)
                name = info.get("name", algo) if info else algo
                print(f"  - {algo:<25} ({name})")
    
    # 显示其他算法
    categorized = []
    for algos in categories.values():
        categorized.extend(algos)
    
    others = [algo for algo in algorithms if algo not in categorized]
    if others:
        print(f"\n其他算法:")
        for algo in others:
            info = get_algorithm_info(algo)
            name = info.get("name", algo) if info else algo
            print(f"  - {algo:<25} ({name})")


def example_different_algorithms():
    """使用不同算法进行对比"""
    print("\n\n" + "=" * 60)
    print("示例4: 多算法对比")
    print("=" * 60)
    
    # 测试图像
    test_image = "test_data/sample_image.png"
    if not os.path.exists(test_image):
        print(f"⚠  测试图像不存在: {test_image}")
        print("   请修改test_image路径为实际图像文件")
        return
    
    # 选择几个代表性算法
    algorithms_to_test = [
        "dinomaly_dinov3_small",
        "mambaad",
        "patchcore"
    ]
    
    print(f"\n测试图像: {test_image}\n")
    print("-" * 60)
    
    results_summary = []
    
    for algo_name in algorithms_to_test:
        try:
            print(f"\n测试算法: {algo_name}")
            
            # 创建检测器
            detector = create_detector(algo_name)
            
            # 加载模型
            detector.load_model()
            
            # 推理
            result = detector.predict(test_image)
            
            # 保存结果
            results_summary.append({
                "algorithm": algo_name,
                "is_anomaly": result.is_anomaly,
                "score": result.anomaly_score,
                "time": result.inference_time
            })
            
            status = "🔴 异常" if result.is_anomaly else "🟢 正常"
            print(f"  结果: {status}")
            print(f"  分数: {result.anomaly_score:.4f}")
            print(f"  时间: {result.inference_time:.2f} ms")
            
            # 释放资源
            detector.release()
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results_summary.append({
                "algorithm": algo_name,
                "is_anomaly": False,
                "score": 0.0,
                "time": 0.0,
                "error": str(e)
            })
    
    # 显示对比总结
    print("\n" + "=" * 60)
    print("算法对比总结:")
    print("=" * 60)
    print(f"{'算法名称':<25} {'结果':<8} {'异常分数':<10} {'推理时间':<10}")
    print("-" * 60)
    
    for result in results_summary:
        algo = result["algorithm"]
        status = "异常" if result["is_anomaly"] else "正常"
        score = f"{result['score']:.4f}"
        time = f"{result['time']:.2f}ms"
        print(f"{algo:<25} {status:<8} {score:<10} {time:<10}")


def example_custom_threshold():
    """自定义阈值示例"""
    print("\n\n" + "=" * 60)
    print("示例5: 动态调整阈值")
    print("=" * 60)
    
    try:
        # 创建检测器
        detector = create_detector("dinomaly_dinov3_small")
        detector.load_model()
        
        test_image = "test_data/sample_image.png"
        if not os.path.exists(test_image):
            print(f"⚠  测试图像不存在: {test_image}")
            detector.release()
            return
        
        # 获取默认阈值下的结果
        default_threshold = detector.threshold
        result_default = detector.predict(test_image)
        
        # 调整阈值并重新推理
        new_threshold = 0.1  # 提高阈值，减少误报
        detector.set_threshold(new_threshold)
        result_new = detector.predict(test_image)
        
        # 显示对比
        print(f"\n测试图像: {test_image}")
        print("\n" + "-" * 50)
        print(f"{'阈值设置':<15} {'是否异常':<8} {'异常分数':<10} {'判定结果':<10}")
        print("-" * 50)
        
        # 默认阈值
        status1 = "🔴 异常" if result_default.is_anomaly else "🟢 正常"
        decision1 = "PASS" if not result_default.is_anomaly else "FAIL"
        print(f"{'默认阈值':<15} {status1:<8} {result_default.anomaly_score:<10.4f} {decision1:<10}")
        
        # 新阈值
        status2 = "🔴 异常" if result_new.is_anomaly else "🟢 正常"
        decision2 = "PASS" if not result_new.is_anomaly else "FAIL"
        print(f"{'阈值=0.1':<15} {status2:<8} {result_new.anomaly_score:<10.4f} {decision2:<10}")
        
        detector.release()
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()


def example_with_device():
    """指定设备示例"""
    print("\n\n" + "=" * 60)
    print("示例6: 指定运行设备")
    print("=" * 60)
    
    # 测试不同设备
    devices_to_test = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    
    test_image = "test_data/sample_image.png"
    if not os.path.exists(test_image):
        print(f"⚠  测试图像不存在: {test_image}")
        return
    
    print(f"\n测试图像: {test_image}\n")
    print("-" * 50)
    
    for device in devices_to_test:
        try:
            print(f"\n设备: {device}")
            
            # 创建设备指定的检测器
            detector = create_detector("dinomaly_dinov3_small", device=device)
            print(f"  实际设备: {detector.device}")
            
            # 加载并推理
            detector.load_model()
            result = detector.predict(test_image)
            
            print(f"  推理时间: {result.inference_time:.2f} ms")
            
            detector.release()
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    print("\n💡 提示: 'auto'会自动选择最快可用设备")


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 60)
    print("统一算法调用示例集")
    print("=" * 60)
    print("\n本示例展示了如何使用统一的接口调用不同的ASD算法\n")
    
    # 检查是否存在测试图像
    test_dir = "test_data"
    if not os.path.exists(test_dir):
        print(f"⚠  测试目录不存在: {test_dir}")
        print("   将创建测试目录，请在其中放置测试图像\n")
        os.makedirs(test_dir, exist_ok=True)
    
    # 运行示例
    print("请选择要运行的示例:")
    print("1. 基础使用 - 单张图像检测")
    print("2. 批量推理")
    print("3. 获取可用算法列表")
    print("4. 多算法对比")
    print("5. 动态调整阈值")
    print("6. 指定运行设备")
    print("0. 运行所有示例")
    print("q. 退出")
    
    choice = input("\n请输入选择 (0-6, q): ").strip().lower()
    
    if choice == 'q':
        print("\n退出示例程序")
        return
    
    # 运行选择的示例
    examples = {
        '1': example_basic_usage,
        '2': example_batch_inference,
        '3': example_algorithm_list,
        '4': example_different_algorithms,
        '5': example_custom_threshold,
        '6': example_with_device
    }
    
    if choice == '0':
        # 运行所有示例
        for i in range(1, 7):
            try:
                examples[str(i)]()
            except Exception as e:
                print(f"\n✗ 示例 {i} 运行出错: {e}")
                import traceback
                traceback.print_exc()
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\n✗ 示例运行出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n无效的选择")
    
    print("\n" + "=" * 60)
    print("示例运行完成")
    print("=" * 60)
    print("\n使用提示:")
    print("1. 修改代码中的 test_image 路径以使用您的图像")
    print("2. 在 config/algorithms.yaml 中配置模型路径")
    print("3. 使用 create_detector() 创建检测器实例")
    print("4. 所有算法都遵循相同的接口规范")
    print("\n")


if __name__ == "__main__":
    # 检查PyTorch是否可用
    try:
        import torch
    except ImportError:
        print("⚠  警告: PyTorch未安装，部分功能可能无法使用")
        print("   请运行: pip install torch torchvision")
        torch = None
    
    main()
