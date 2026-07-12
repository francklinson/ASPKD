"""
算法可用性动态检查模块

在服务启动时自动检查:
1. Python 库依赖是否可导入
2. 预训练模型文件是否存在
3. 训练脚本是否存在

根据检查结果动态决定每个算法的推理/训练可用性，替代静态 EXCLUDED_ALGORITHMS 集合。
"""

import os
import sys
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class AlgorithmAvailability:
    """单个算法的可用性状态"""
    algorithm_id: str
    family: str = ""
    inference_available: bool = False
    training_available: bool = False
    missing_libraries: List[str] = field(default_factory=list)
    missing_model_files: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


# ============================================================================
# 模块级缓存
# ============================================================================

_availability_cache: Dict[str, AlgorithmAvailability] = {}
_cache_populated: bool = False
_project_root: str = ""


def is_cache_populated() -> bool:
    """检查可用性缓存是否已初始化"""
    return _cache_populated


def get_all_availability() -> Dict[str, AlgorithmAvailability]:
    """获取所有算法的可用性缓存"""
    return _availability_cache


def get_algorithm_availability(alg_id: str) -> Optional[AlgorithmAvailability]:
    """获取单个算法的可用性"""
    return _availability_cache.get(alg_id)


def get_available_algorithms() -> List[str]:
    """获取所有推理可用的算法 ID 列表"""
    return sorted([
        alg_id for alg_id, avail in _availability_cache.items()
        if avail.inference_available
    ])


# ============================================================================
# 算法家族定义
# ============================================================================

# 算法 → 家族 映射（优先从 ALGORITHM_GROUPS 构建，运行时自动补充）
_algorithm_family_map: Dict[str, str] = {}

# 家族 → 训练可用性判断规则
FAMILY_TRAINING_RULES: Dict[str, str] = {
    "Dinomaly": "script",          # 需要训练脚本存在
    "Dinomaly2 (预览)": "script",
    "Anomalib": "script",          # 需要训练脚本 + anomalib 本地源码
    "ADer": "script",              # 需要训练脚本存在
    "MuSc (零样本)": "never",       # 零样本检测，设计上不训练
    "SubspaceAD (少样本)": "never",  # 少样本检测，设计上不训练
    "BaseASD": "never",            # TensorFlow 依赖，永久不可训练
    "Stubs": "never",              # 空壳实现
    "Other": "never",
}

# 家族 → 必需 Python 库（库名 → 描述）
FAMILY_LIBRARY_REQUIREMENTS: Dict[str, Dict[str, str]] = {
    "Dinomaly": {
        "torch": "PyTorch",
        "timm": "PyTorch Image Models",
    },
    "Dinomaly2 (预览)": {
        "torch": "PyTorch",
        "timm": "PyTorch Image Models",
    },
    "Anomalib": {
        "torch": "PyTorch",
    },
    "ADer": {
        "torch": "PyTorch",
        "timm": "PyTorch Image Models",
        "fvcore": "fvcore (ADer 模型组件)",
        "tensorboardX": "tensorboardX (ADer 训练日志)",
    },
    "MuSc (零样本)": {
        "torch": "PyTorch",
    },
    "SubspaceAD (少样本)": {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
    },
    "BaseASD": {
        "tensorflow": "TensorFlow",
        "keras": "Keras",
    },
    "Stubs": {},
    "Other": {},
}

# 特定算法的额外库依赖（在家族依赖之上叠加）
ALGORITHM_EXTRA_LIBS: Dict[str, Dict[str, str]] = {
    "mambaad": {
        "mamba_ssm": "MambaAD 状态空间模型 (CUDA 扩展)",
        "causal_conv1d": "MambaAD 因果卷积 (CUDA 扩展)",
    },
    "winclip": {
        "open_clip": "WinClip 零样本检测",
    },
    # MuSc CLIP 变体需要 open_clip（vendored，但检查系统级安装优先）
    "musc_clip_b32_512": {"open_clip": "OpenCLIP"},
    "musc_clip_b16_512": {"open_clip": "OpenCLIP"},
    "musc_clip_l14_336": {"open_clip": "OpenCLIP"},
    "musc_clip_l14_518": {"open_clip": "OpenCLIP"},
}

# 已知的空壳/存根算法（load_model 直接抛出 NotImplementedError）
PERMANENTLY_UNAVAILABLE: set = {
    "hiad", "multiads", "dictas",          # 空壳存根
    "musc", "subspacead",                   # 旧版统一入口（已废弃）
    "audio_feature_cluster",                # 音频专用，不适用图片检测
    "denseae", "cae", "vae", "aegan", "differnet",  # BaseASD (TensorFlow)
}

# 已知的零样本/少样本算法（无需训练）
ZERO_OR_FEW_SHOT_PREFIXES: Tuple[str, ...] = (
    "musc_clip_", "musc_dinov2_", "subspacead_dinov2_",
)

# 不需要预训练 checkpoint 即可推理的算法
# - padim/dfkde: 特征建模类，直接从预训练 backbone 建模，无需训练
# - efficient_ad: 需要 teacher 权重（单独检查），不需要 trained checkpoint
# - Anomalib v2.5.0 新增算法: 使用 .placeholder 路径，模型内部初始化
CHECKPOINT_FREE_INFERENCE: set = {
    # 特征建模类 — 直接从预训练 backbone 建模
    "padim", "dfkde",
    # 需要额外依赖但不需要 trained checkpoint
    "efficient_ad", "fastflow", "stfpm", "uflow", "cfm", "vlm_ad",
}

# 不可训练的算法（即使训练脚本存在，这些算法设计上/依赖上不可训练）
# 与 training.py ALGORITHM_FAMILIES 中的 trainable: False 保持一致
NON_TRAINABLE_ALGORITHMS: set = {
    # Anomalib — 无需训练 / 依赖缺失
    "padim", "dfkde", "efficient_ad",
    "fastflow", "stfpm", "uflow", "cfm", "vlm_ad",
    # ADer — 部分算法仅推理
    # BaseASD — 全部不可训练（已在 PERMANENTLY_UNAVAILABLE 中）
}

# 训练脚本路径（相对于项目根目录）
TRAINING_SCRIPTS: Dict[str, str] = {
    "Dinomaly": "algorithms/Dinomaly/dinomaly_train_evaluate.py",
    "Dinomaly2 (预览)": "algorithms/Dinomaly2/dinomaly_2D.py",
    "Anomalib": "algorithms/anomalib/engine/engine.py",  # anomalib 本地源码
    "ADer": "algorithms/ADer/run.py",
}


# ============================================================================
# 工具函数
# ============================================================================

def _resolve_project_root() -> str:
    """解析项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_family_map(all_alg_ids: List[str]) -> Dict[str, str]:
    """从 ALGORITHM_GROUPS 构建算法→家族映射，未匹配的归入 Stubs 或 Other"""
    try:
        from backend.api.custom_detection import ALGORITHM_GROUPS
        for group, algs in ALGORITHM_GROUPS.items():
            for alg in algs:
                _algorithm_family_map[alg] = group
    except ImportError:
        pass

    # 补充未在 ALGORITHM_GROUPS 中的算法
    for alg_id in all_alg_ids:
        if alg_id not in _algorithm_family_map:
            if alg_id in PERMANENTLY_UNAVAILABLE:
                _algorithm_family_map[alg_id] = "Stubs" if alg_id not in ("denseae", "cae", "vae", "aegan", "differnet") else "BaseASD"
            else:
                _algorithm_family_map[alg_id] = "Other"

    return _algorithm_family_map


def _check_import(module_name: str) -> bool:
    """检查模块是否可导入（使用 importlib.util.find_spec，不实际加载以避免副作用）"""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _check_anomalib_available(project_root: str) -> bool:
    """
    检查 anomalib 本地源码是否可用。
    algorithms/anomalib/ 是 anomalib v2.5.0 库源码，algorithms/ 在 sys.path 中。
    """
    anomalib_dir = os.path.join(project_root, "algorithms", "anomalib")
    if not os.path.isdir(anomalib_dir):
        return False
    # 检查关键子模块是否存在
    models_dir = os.path.join(anomalib_dir, "models")
    engine_dir = os.path.join(anomalib_dir, "engine")
    init_file = os.path.join(anomalib_dir, "__init__.py")
    return (os.path.isdir(models_dir) and os.path.isdir(engine_dir)
            and os.path.isfile(init_file))


def _resolve_model_path_from_config(
    config: dict, algorithm_name: str, project_root: str
) -> Optional[str]:
    """
    从 config.yaml 解析算法的模型路径。
    复用与 factory._resolve_model_path 相同的逻辑。
    返回绝对路径，或 None（如果路径是 .placeholder 或不存在于配置中）。
    """
    models_cfg = config.get("models", {})
    if not models_cfg:
        return None

    # 尝试1: 完整名称拆分（如 dinomaly_dinov3_small → algo=dinomaly, variant=dinov3_small）
    variants_to_try = [
        'dinov3_small', 'dinov3_base', 'dinov3_large',
        'dinov2_small', 'dinov2_base', 'dinov2_large',
        'small', 'large', 'base', 'default',
    ]
    for variant in variants_to_try:
        if algorithm_name.endswith(f'_{variant}'):
            algo = algorithm_name[:-len(f'_{variant}')]
            algo_cfg = models_cfg.get(algo)
            if algo_cfg and isinstance(algo_cfg, dict):
                path = algo_cfg.get(variant)
                if path:
                    return _to_absolute_path(path, project_root)

    # 尝试2: 按 '_' 分成两部分
    parts = algorithm_name.split('_', 1)
    if len(parts) == 2:
        algo, variant = parts
        algo_cfg = models_cfg.get(algo)
        if algo_cfg and isinstance(algo_cfg, dict):
            path = algo_cfg.get(variant)
            if path:
                return _to_absolute_path(path, project_root)

    # 尝试3: 直接作为算法名，variant=default
    algo_cfg = models_cfg.get(algorithm_name)
    if algo_cfg and isinstance(algo_cfg, dict):
        path = algo_cfg.get("default")
        if path:
            return _to_absolute_path(path, project_root)

    # 尝试4: 完整名称作为 variant，遍历常见算法名
    common_algos = [
        'dinomaly', 'mambaad', 'invad', 'vitad', 'unad',
        'cflow', 'pyramidflow', 'simplenet', 'patchcore',
        'efficient_ad', 'padim',
        'cfa', 'csflow', 'dfkde', 'dfm', 'draem', 'dsr', 'fastflow', 'fre',
        'reverse_distillation', 'stfpm', 'ganomaly', 'supersimplenet',
        'uflow', 'uninet', 'vlm_ad', 'winclip',
        'anomalyvfm', 'cfm', 'general_ad', 'glass', 'inp_former',
        'l2bt', 'patchflow', 'anomaly_dino',
        'destseg', 'realnet', 'rdpp',
        'hiad', 'multiads', 'dictas', 'audio_feature_cluster',
        'denseae', 'cae', 'vae', 'aegan', 'differnet',
    ]
    for algo in common_algos:
        algo_cfg = models_cfg.get(algo)
        if algo_cfg and isinstance(algo_cfg, dict):
            path = algo_cfg.get(algorithm_name)
            if path:
                return _to_absolute_path(path, project_root)

    return None


def _to_absolute_path(path: str, project_root: str) -> str:
    """将相对路径转为绝对路径（相对于项目根目录）"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


def _is_placeholder(path: str) -> bool:
    """判断路径是否是占位符（.placeholder 表示未配置）"""
    return path.endswith('/.placeholder') or path.endswith('\\.placeholder') or path == '.placeholder'


# ============================================================================
# 核心检查逻辑
# ============================================================================

def _check_single_algorithm(
    alg_id: str,
    family: str,
    config: dict,
    project_root: str,
    lib_cache: Dict[str, bool],
    anomalib_available: Optional[bool],
) -> AlgorithmAvailability:
    """检查单个算法的可用性"""

    result = AlgorithmAvailability(algorithm_id=alg_id, family=family)

    # ---- 1. 永久不可用检查 ----
    if alg_id in PERMANENTLY_UNAVAILABLE:
        result.reasons.append("永久不可用（空壳存根/废弃入口/不适用）")
        return result

    # ---- 2. 库依赖检查 ----
    # 2a. 家族级依赖
    family_libs = FAMILY_LIBRARY_REQUIREMENTS.get(family, {})
    for lib, desc in family_libs.items():
        if lib not in lib_cache:
            # Anomalib 特殊处理：检查本地源码而非 pip 包
            if family == "Anomalib" and lib == "torch":
                # torch 是基础依赖，单独检查
                lib_cache[lib] = _check_import(lib)
            else:
                lib_cache[lib] = _check_import(lib)
        if not lib_cache[lib]:
            result.missing_libraries.append(lib)
            result.reasons.append(f"缺少库: {lib} ({desc})")

    # 2b. 算法级额外依赖
    extra_libs = ALGORITHM_EXTRA_LIBS.get(alg_id, {})
    for lib, desc in extra_libs.items():
        if lib not in lib_cache:
            lib_cache[lib] = _check_import(lib)
        if not lib_cache[lib]:
            result.missing_libraries.append(lib)
            result.reasons.append(f"缺少库: {lib} ({desc})")

    # Anomalib 特殊检查
    if family == "Anomalib":
        if anomalib_available is None:
            anomalib_available = _check_anomalib_available(project_root)
        if not anomalib_available:
            result.missing_libraries.append("anomalib (本地源码)")
            result.reasons.append("缺少库: anomalib (本地源码 algorithms/anomalib/ 不存在或不完整)")

    # ---- 3. 推理可用性判断 ----
    result.inference_available = len(result.missing_libraries) == 0

    # ---- 4. 模型文件检查 ----
    # 跳过不需要 checkpoint 的算法（特征建模类 / .placeholder 路径）
    model_path = _resolve_model_path_from_config(config, alg_id, project_root)
    skip_checkpoint_check = (
        alg_id in CHECKPOINT_FREE_INFERENCE
        or alg_id.startswith(ZERO_OR_FEW_SHOT_PREFIXES)
    )
    if model_path and not _is_placeholder(model_path) and not skip_checkpoint_check:
        if not os.path.exists(model_path):
            rel_path = os.path.relpath(model_path, project_root)
            result.missing_model_files.append(rel_path)
            result.reasons.append(f"缺少模型文件: {rel_path}")
            result.inference_available = False

    # ---- 5. 训练可用性判断 ----
    if alg_id in NON_TRAINABLE_ALGORITHMS:
        result.training_available = False
    else:
        rule = FAMILY_TRAINING_RULES.get(family, "never")
        if rule == "never":
            result.training_available = False
        elif rule == "script":
            script_rel = TRAINING_SCRIPTS.get(family, "")
            if script_rel:
                script_abs = os.path.join(project_root, script_rel)
                result.training_available = os.path.exists(script_abs)
            else:
                result.training_available = False

            # 训练也需要基础库可用
            if result.training_available and result.missing_libraries:
                result.training_available = False

    # 零样本/少样本：推理可用即为"完全可用"
    if alg_id.startswith(ZERO_OR_FEW_SHOT_PREFIXES):
        result.training_available = False  # 设计上不训练

    return result


def check_all_algorithms(config_path: Optional[str] = None) -> Dict[str, AlgorithmAvailability]:
    """
    运行完整的算法可用性检查。
    首次调用会填充缓存，后续调用返回缓存。

    Args:
        config_path: config.yaml 路径，默认使用项目标准路径

    Returns:
        {algorithm_id: AlgorithmAvailability}
    """
    global _availability_cache, _cache_populated, _project_root

    if _cache_populated:
        return _availability_cache

    project_root = _resolve_project_root()
    _project_root = project_root

    # 确保 algorithms/ 在 sys.path 中（适配器模块依赖此路径）
    algorithms_dir = os.path.join(project_root, "algorithms")
    if algorithms_dir not in sys.path:
        sys.path.insert(0, algorithms_dir)

    # 加载配置
    if config_path is None:
        config_path = os.path.join(project_root, "backend", "config", "config.yaml")

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[AlgorithmAvailability] 加载配置失败: {e}")
        return {}

    # 获取所有已注册算法
    try:
        from algorithms.factory import _import_adapters
        _import_adapters()
        from backend.core import AlgorithmRegistry
        all_alg_ids = AlgorithmRegistry.list_algorithms()
    except Exception as e:
        print(f"[AlgorithmAvailability] 获取算法列表失败: {e}")
        return {}

    # 构建家族映射
    family_map = _build_family_map(all_alg_ids)

    # 检查 anomalib 本地源码
    anomalib_available = _check_anomalib_available(project_root)

    # 运行检查
    lib_cache: Dict[str, bool] = {}
    results: Dict[str, AlgorithmAvailability] = {}

    print(f"[AlgorithmAvailability] 正在检查 {len(all_alg_ids)} 个算法...")
    for alg_id in sorted(all_alg_ids):
        family = family_map.get(alg_id, "Other")
        results[alg_id] = _check_single_algorithm(
            alg_id, family, config, project_root, lib_cache, anomalib_available
        )

    _availability_cache = results
    _cache_populated = True
    return results


def initialize(config_path: Optional[str] = None) -> Dict[str, AlgorithmAvailability]:
    """
    启动时初始化（等同于 check_all_algorithms + 打印摘要）。
    由 start_server.py 或 backend/main.py 的 lifespan 调用。
    """
    results = check_all_algorithms(config_path)

    # 统计
    total = len(results)
    avail_inference = sum(1 for r in results.values() if r.inference_available)
    avail_training = sum(1 for r in results.values() if r.training_available)

    print(f"[AlgorithmAvailability] 检查完成: {total} 个算法")
    print(f"  推理可用: {avail_inference}/{total}")
    print(f"  训练可用: {avail_training}/{total}")

    # 按家族统计
    family_stats: Dict[str, Dict[str, int]] = {}
    for r in results.values():
        fam = r.family or "Other"
        if fam not in family_stats:
            family_stats[fam] = {"total": 0, "inference": 0, "training": 0}
        family_stats[fam]["total"] += 1
        if r.inference_available:
            family_stats[fam]["inference"] += 1
        if r.training_available:
            family_stats[fam]["training"] += 1

    for fam, stats in sorted(family_stats.items()):
        inf = stats["inference"]
        tot = stats["total"]
        tr = stats["training"]
        if inf == tot:
            print(f"  ✓ {fam}: {inf}/{tot} 推理可用")
        elif inf > 0:
            print(f"  ⚠ {fam}: {inf}/{tot} 推理可用 ({tot - inf} 不可用)")
        else:
            print(f"  ✗ {fam}: 0/{tot} 推理可用 - 全部不可用")
        if tr > 0:
            print(f"        训练可用: {tr} 个")

    # 列出不可用算法
    unavailable = [(alg_id, r) for alg_id, r in results.items() if not r.inference_available]
    if unavailable:
        print(f"\n  不可用算法 ({len(unavailable)} 个):")
        for alg_id, r in unavailable:
            reason = r.reasons[0] if r.reasons else "未知原因"
            print(f"    • {alg_id}: {reason}")

    return results
