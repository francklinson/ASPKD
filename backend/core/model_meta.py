"""
模型元数据推断 - 从模型目录/文件名推断算法族、算法名等信息
"""
import os
import json
from typing import Optional, Dict

# ADer 训练器名 → 算法ID 映射
_ADER_TRAINER_MAP = {
    "MAMBAADTrainer": "mambaad",
    "InvADTrainer": "invad",
    "ViTADTrainer": "vitad",
    "UniADTrainer": "unad",
    "CFLOWTrainer": "cflow",
    "PyramidFlowTrainer": "pyramidflow",
    "SimpleNetTrainer": "simplenet",
    "DeSTSegTrainer": "destseg",
    "RealNetTrainer": "realnet",
    "RDPTrainer": "rdpp",
}


def infer_model_meta(name: str, base_dir: str = "") -> Dict[str, str]:
    """从模型名推断算法族、算法名等信息

    优先读取同目录下的 metadata.json，其次按命名规则推断。

    Args:
        name: 模型目录名或文件名
        base_dir: 模型所在目录（用于读取 metadata.json）

    Returns:
        dict with keys: algorithm_family, algorithm_name, model_type, model_size, category, data_source
    """
    result = {
        "algorithm_family": "",
        "algorithm_name": "",
        "model_type": "",
        "model_size": "",
        "category": "",
        "data_source": "",
    }

    # 1. 尝试从 metadata.json 读取（最可靠）
    if base_dir:
        meta_path = os.path.join(base_dir, name, "metadata.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(base_dir, name.rstrip(".pth"), "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for k in result:
                    if k in saved and saved[k]:
                        result[k] = saved[k]
                return result
            except Exception:
                pass

    fname = name.lower()

    # 2. 按前缀判断算法族
    if fname.startswith("anomalib_"):
        result["algorithm_family"] = "anomalib"
        parts = name.split("_")
        if len(parts) >= 2:
            result["algorithm_name"] = parts[1]
    elif fname.startswith("dinomaly2_"):
        result["algorithm_family"] = "dinomaly2"
        result["model_type"] = "dinov3" if "dinov3" in fname else "dinov2"
        result["model_size"] = _infer_size(fname)
        result["algorithm_name"] = f"{result['model_type']}_{result['model_size']}"
    elif fname.startswith("dinomaly_"):
        result["algorithm_family"] = "dinomaly"
        result["model_type"] = "dinov3" if "dinov3" in fname else "dinov2"
        result["model_size"] = _infer_size(fname)
        result["algorithm_name"] = f"{result['model_type']}_{result['model_size']}"
    elif _is_ader_model(name):
        result["algorithm_family"] = "ader"
        result["algorithm_name"] = _infer_ader_algo(name)
    elif "anomalib" in fname:
        result["algorithm_family"] = "anomalib"
    elif "ader" in fname:
        result["algorithm_family"] = "ader"

    # 3. 推断训练类别和数据来源
    _infer_category_source(name, result)

    return result


def save_model_meta(base_dir: str, name: str, **kwargs) -> None:
    """保存模型元数据到 metadata.json"""
    meta_path = os.path.join(base_dir, name, "metadata.json")
    try:
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(kwargs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _is_ader_model(name: str) -> bool:
    """判断是否为 ADer 框架训练的模型"""
    # ADer 命名模式: {TrainerName}Trainer_configs_benchmark_...
    for trainer_name in _ADER_TRAINER_MAP:
        if name.startswith(trainer_name):
            return True
    # 通用模式: 以 *Trainer_configs_ 开头
    if "Trainer_configs_" in name:
        return True
    return False


def _infer_ader_algo(name: str) -> str:
    """从 ADer 模型目录名推断算法名"""
    for trainer_name, algo_id in _ADER_TRAINER_MAP.items():
        if name.startswith(trainer_name):
            return algo_id
    # 通用: 从 configs_benchmark_ 后提取
    if "configs_benchmark_" in name:
        after = name.split("configs_benchmark_", 1)[1]
        # 第一个下划线前的部分是算法名
        algo = after.split("_")[0].lower()
        return algo
    return ""


def _infer_size(fname: str) -> str:
    """从文件名推断模型大小"""
    if "large" in fname:
        return "large"
    elif "base" in fname:
        return "base"
    return "small"


def _infer_category_source(name: str, result: Dict) -> None:
    """从模型名推断训练类别和数据来源"""
    parts = name.replace("-", "_").split("_")
    source_keywords = {"mvtec", "visa", "spk"}
    known_categories = {
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
    }

    for i, p in enumerate(parts):
        pl = p.lower()
        if pl in source_keywords:
            result["data_source"] = pl
            if i > 0:
                candidate = parts[i - 1].lower()
                exclude = {
                    "dinomaly", "dinomaly2", "anomalib", "ader",
                    "dinov2", "dinov3", "small", "base", "large",
                    result.get("algorithm_name", "").lower(),
                }
                if candidate not in exclude:
                    result["category"] = parts[i - 1]
            break

    if not result["category"]:
        for p in parts:
            if p.lower() in known_categories:
                result["category"] = p
                if not result["data_source"]:
                    if p.lower() in {"bottle", "cable", "capsule", "carpet", "grid",
                                     "hazelnut", "leather", "metal_nut", "pill", "screw",
                                     "tile", "toothbrush", "transistor", "wood", "zipper"}:
                        result["data_source"] = "mvtec"
                    else:
                        result["data_source"] = "spk"
                break
