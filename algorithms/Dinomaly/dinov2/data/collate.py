import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    """
    整合数据并转换为指定数据类型的函数

    参数:
        samples_list: 样本列表，每个样本包含全局和局部裁剪
        mask_ratio_tuple: 掩码比例的元组，定义掩码范围
        mask_probability: 掩码概率，决定样本中被掩码的比例
        dtype: 目标数据类型
        n_tokens: 令牌数量，用于掩码生成
        mask_generator: 掩码生成器函数

    返回:
        包含整合后的全局和局部裁剪、掩码等相关信息的字典
    """

    n_global_crops = len(samples_list[0][0]["global_crops"])  # 获取全局裁剪的数量
    n_local_crops = len(samples_list[0][0]["local_crops"])  # 获取局部裁剪的数量

    # 整合所有样本的全局裁剪，堆叠成一个张量
    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    # 整合所有样本的局部裁剪，堆叠成一个张量
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)  # 获取批次大小
    N = n_tokens  # 获取令牌数量
    n_samples_masked = int(B * mask_probability)  # 计算需要掩码的样本数量
    # 生成一个从mask_ratio_tuple开始到结束的均匀分布概率数组
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    # 为需要掩码的样本生成掩码
    for i in range(0, n_samples_masked):
        prob_min = probs[i]  # 当前掩码比例的最小值
        prob_max = probs[i + 1]  # 当前掩码比例的最大值
        # 根据随机生成的掩码比例创建掩码，并添加到掩码列表
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)  # 更新掩码上界
    # 对于不需要掩码的样本，创建空掩码
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)  # 随机打乱掩码列表

    # 将掩码列表堆叠成一个张量，并展平
    collated_masks = torch.stack(masks_list).flatten(1)
    # 获取被掩码的索引
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    # 计算掩码权重，考虑掩码的总数
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    # 返回包含处理结果的字典
    return {
        "collated_global_crops": collated_global_crops.to(dtype),  # 转换为指定数据类型
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,  # 转换为指定数据类型
        "mask_indices_list": mask_indices_list,  # 整合后的掩码
        "masks_weight": masks_weight,  # 被掩码的索引列表
        "upperbound": upperbound,  # 掩码权重
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),  # 掩码上界
    }  # 被掩码的补丁数量
