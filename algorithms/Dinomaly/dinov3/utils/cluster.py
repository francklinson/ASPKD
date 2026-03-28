import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


# 定义一个枚举类ClusterType，用于表示集群类型
class ClusterType(Enum):
    # 定义CW类型的集群，值为"cw"
    CW = "cw"


def _guess_cluster_type() -> ClusterType:
    """
    猜测集群类型并返回对应的集群类型枚举值

    Returns:
        ClusterType: 返回集群类型的枚举值，当前固定返回ClusterType.CW
    """
    return ClusterType.CW  # 直接返回集群类型CW


def get_cluster_type(
        cluster_type: Optional[ClusterType] = None,
) -> Optional[ClusterType]:
    """
    获取集群类型

    该函数用于获取集群类型，如果未提供集群类型参数，则会尝试自动猜测集群类型。

    参数:
        cluster_type (Optional[ClusterType]): 可选的集群类型参数，如果未提供则为None

    返回:
        Optional[ClusterType]: 返回集群类型，如果无法确定则返回None
    """
    if cluster_type is None:  # 如果未提供集群类型参数
        return _guess_cluster_type()  # 调用内部函数尝试自动猜测集群类型

    return cluster_type  # 直接返回提供的集群类型


def get_slurm_account(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    """
    根据集群类型获取对应的SLURM账户名称
    参数:
        cluster_type: 集群类型，可选参数，如果不提供则自动获取当前集群类型
    返回:
        对应的SLURM账户名称字符串，如果集群类型不支持则返回None
    """
    # 如果未提供集群类型，则获取当前集群类型
    cluster_type = get_cluster_type(cluster_type)
    # 如果集群类型为None，直接返回None
    if cluster_type is None:
        return None
    # 根据集群类型返回对应的SLURM账户名称
    return {
        ClusterType.CW: "fair_amaia_cw_explore",
    }[cluster_type]


def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    """
    根据集群类型获取检查点文件的路径
    参数:
        cluster_type: 集群类型，可选参数，默认为None
    返回:
        Optional[Path]: 如果集群类型有效则返回对应的检查点路径，否则返回None
    """
    # 如果未提供集群类型，则获取默认的集群类型
    cluster_type = get_cluster_type(cluster_type)
    # 如果集群类型为None，直接返回None
    if cluster_type is None:
        return None

    # 定义不同集群类型对应的检查点目录名
    CHECKPOINT_DIRNAMES = {
        ClusterType.CW: "",
    }
    # 构造并返回完整的检查点路径
    return Path("/") / CHECKPOINT_DIRNAMES[cluster_type]


def get_user_checkpoint_path(
        cluster_type: Optional[ClusterType] = None,
) -> Optional[Path]:
    """
    获取用户特定的检查点路径
    该函数根据集群类型获取基础检查点路径，然后结合当前用户名
    形成用户特定的检查点路径。如果基础检查点路径不存在，则返回None。
    参数:
        cluster_type: 集群类型，可选参数。如果未提供，则使用默认配置
    返回:
        Optional[Path]: 用户特定的检查点路径，如果基础检查点路径不存在则返回None
    异常:
        AssertionError: 当无法获取当前用户名时触发
    """
    # 获取基础检查点路径
    checkpoint_path = get_checkpoint_path(cluster_type)
    # 如果基础路径不存在，直接返回None
    if checkpoint_path is None:
        return None

    # 获取当前系统用户名
    username = os.environ.get("USER")
    # 断言确保用户名不为空
    assert username is not None
    # 拼接用户路径并返回
    return checkpoint_path / username


def get_slurm_qos(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    """
    根据集群类型获取对应的SLURM QoS值
    参数:
        cluster_type: 集群类型，可选参数，默认为None
    返回:
        返回对应的SLURM QoS字符串，如果集群类型不存在则返回None
    """
    # 获取集群类型，如果未提供则使用默认值
    cluster_type = get_cluster_type(cluster_type)
    # 如果集群类型为None，直接返回None
    if cluster_type is None:
        return None

    # 使用字典映射集群类型到对应的QoS值
    return {
        ClusterType.CW: "explore",
    }.get(cluster_type)


def get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    """
    根据集群类型获取对应的SLURM分区名称
    参数:
        cluster_type: 集群类型枚举，如果为None则自动获取当前集群类型
    返回:
        str: 对应的SLURM分区名称，如果无法确定则返回None
    """
    # 确保获取到有效的集群类型
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    # 定义不同集群类型对应的SLURM分区映射关系
    SLURM_PARTITIONS = {
        ClusterType.CW: "learn",
    }
    # 返回对应集群类型的SLURM分区名称
    return SLURM_PARTITIONS[cluster_type]


def get_slurm_executor_parameters(
        nodes: int,
        num_gpus_per_node: int,
        cluster_type: Optional[ClusterType] = None,
        **kwargs,
) -> Dict[str, Any]:
    # create default parameters
    params = {
        "mem_gb": 0,  # Requests all memory on a node, see https://slurm.schedmd.com/sbatch.html
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        "cpus_per_task": 10,
        "nodes": nodes,
        "slurm_partition": get_slurm_partition(cluster_type),
    }
    # apply cluster-specific adjustments
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type == ClusterType.CW:
        params["cpus_per_task"] = 16
    # set additional parameters / apply overrides
    params.update(kwargs)
    return params
