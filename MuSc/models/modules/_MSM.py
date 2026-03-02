import torch
from tqdm import tqdm

"""
translated by codegeex:
这段代码实现了一种基于**最近邻搜索**的异常检测算法，通常用于无监督异常检测任务，特别是在工业缺陷检测（如AD算法中的PaDiM、PatchCore等）中。
代码的核心是计算每个图像中每个特征块与数据集中其他图像中对应特征块的距离，通过统计这些距离来生成异常分数。

以下是对代码的详细解释：

### 1. 核心概念与实现原理

该算法的核心思想是：**如果一个图像中的某个特征块在训练集中很难找到相似的特征块（即距离都很远），那么这个特征块很可能是异常的。**

代码中包含两个主要的计算函数 `compute_scores_fast` 和 `compute_scores_slow`，以及一个封装函数 `MSM`。

*   **输入 `Z`**:
    通常是一个形状为 `(图像数量, 每张图像的块数量, 特征维度)` 的张量。
    例如：`Z = torch.rand(200, 1369, 1024)` 表示有200张图，每张图被切分为1369个patch，每个patch的特征向量长度为1024。

*   **计算逻辑**:
    对于第 `i` 张图的每一个特征块，算法会遍历其他所有图（`j != i`），找到其他图中对应位置的特征块，并计算它们之间的距离（通常使用欧氏距离 `torch.cdist`）。
    对于每个位置，算法会记录下该位置在所有其他图中的最小距离。
    最终，对于第 `i` 张图，我们会得到一组距离值（每个patch一个最小距离）。

*   **区间平均**:
    得到距离集合后，并不是直接取平均值，而是进行了一种类似“截尾平均”的操作：
    1.  利用 `torch.topk(..., largest=False)` 找出距离最小的 `k_max` 个值。
    2.  再从这 `k_max` 个值中，利用 `torch.topk(..., largest=True)` 去掉距离最小的 `k_min` 个值（即保留中间较大的部分）。
    3.  对剩下的值求平均。
    这种做法可以过滤掉极个别的非常相似的情况（可能是噪声），专注于“稍微有点远但不是特别远”的那些特征，或者根据具体参数调整对异常的敏感度。

### 2. 函数详解

#### `compute_scores_fast(Z, i, device, ...)`
*   **原理**: **速度快，显存占用大**。
*   **实现**:
    *   它利用矩阵运算一次性计算第 `i` 张图与所有其他图（`Z_ref`）的距离。
    *   `torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c))`：这行代码计算了当前图 `i` 的所有patch与其他所有图的所有patch之间的距离矩阵。
    *   通过 `reshape` 和 `min` 操作快速提取出每个patch位置在所有其他参考图中的最小距离。
*   **优点**: 利用了GPU的并行计算能力，计算速度非常快。
*   **缺点**: 需要构建一个巨大的距离矩阵（`patch_num * (image_num-1) * patch_num`），如果图片数量很多或patch数量很多，显存容易溢出（OOM）。

#### `compute_scores_slow(Z, i, device, ...)`
*   **原理**: **速度慢，显存占用小**。
*   **实现**:
    *   使用 `for` 循环逐张处理参考图 `j`。
    *   每次只计算图 `i` 和图 `j` 之间的距离，并立即取最小值，将结果拼接到结果张量中。
*   **优点**: 显存占用极低，因为不需要存储巨大的距离矩阵，只需要存储中间结果。
*   **缺点**: Python循环在GPU上效率较低，且频繁的 `torch.cat` 操作也会导致性能下降。

#### `MSM(Z, device, ...)`
*   **用途**: 封装函数，用于计算整个数据集 `Z` 的异常分数矩阵。
*   **流程**:
    1.  初始化一个空张量 `anomaly_scores_matrix`。
    2.  遍历数据集中的每一张图 `i`。
    3.  调用 `compute_scores_fast`（默认）计算第 `i` 张图的异常分数。
    4.  将结果拼接到总矩阵中。
*   **输出**: 返回形状为 `(N, B)` 的矩阵，其中 `N` 是图像数量，`B` 是每张图的patch数量（代码中注释写的是B，实际对应patch_num）。这个矩阵中的数值越大，表示对应位置的异常程度越高。

### 3. 参数解释

*   `topmin_max` (默认 0.3): 选取最近邻的最大比例。如果是浮点数且小于1，则表示选取前 30% 的最小距离；如果是整数，则表示选取具体的个数。
*   `topmin_min` (默认 0): 选取最近邻的最小比例。用于在选出的最近邻中，剔除距离最小的那一部分（即最相似的）。
*   **逻辑**:
    *   假设有100个参考图。
    *   `topmin_max=0.3` -> 选取距离最近的30个图。
    *   `topmin_min=0` -> 从这30个图中，去掉距离最近的0个（即保留全部30个）。
    *   如果 `topmin_min=0.1` -> 从这30个图中，再去掉距离最近的10%（即3个），最终对剩下的27个距离求平均。
"""

"""
We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""


def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i + 1:]), dim=0)
    patch2image = torch.cdist(Z[i:i + 1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num - 1, patch_num)
    patch2image = torch.min(patch2image, -1)[0]
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1] * k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1] * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)


def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    """
    计算异常分数的函数，使用较慢的方法但节省空间
    参数:
        Z: 输入的张量，形状为 [batch_size, feature_dim]
        i: 当前要处理的样本索引
        device: 计算设备（CPU或GPU）
        topmin_min: 选取最近邻的最小比例，默认为0
        topmin_max: 选取最近邻的最大比例，默认为0.3
    返回:
        torch.mean(patch2image, dim=1): 计算得到的异常分数
    """
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1] * k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1] * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)


def MSM(Z, device, topmin_min=0, topmin_max=0.3):
    """
    计算异常分数矩阵的函数
    参数:
        Z: 输入数据张量
        device: 计算设备 (CPU/GPU)
        topmin_min: 最小值阈值，默认为0
        topmin_max: 最大值阈值，默认为0.3
    返回:
        anomaly_scores_matrix: 异常分数矩阵，形状为(N, B)，其中N是样本数量，B是批次大小
    """
    # 初始化一个空的张量用于存储异常分数，并将其移动到指定设备上
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    # 使用tqdm进度条遍历所有样本
    for i in tqdm(range(Z.shape[0])):
        # for i in range(Z.shape[0]):  # 这行被注释掉了，可能是用于调试的原始代码
        # 计算第i个样本的异常分数，并增加一个维度
        anomaly_scores_i = compute_scores_fast(Z, i, device, topmin_min, topmin_max).unsqueeze(0)
        # 将当前样本的异常分数矩阵与总矩阵拼接起来
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)  # (N, B)
    return anomaly_scores_matrix


if __name__ == "__main__":
    device = 'cuda:0'
    import time

    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time - s_time) * 1000)
