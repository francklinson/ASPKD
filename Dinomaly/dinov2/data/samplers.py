import itertools  # 导入itertools模块，用于创建高效的循环
import warnings  # 导入warnings模块，用于处理警告信息
from typing import Any, Optional  # 导入类型提示相关的类

import numpy as np  # 导入NumPy库，用于科学计算
import torch  # 导入PyTorch库，用于深度学习
from torch.utils.data.sampler import Sampler  # 导入PyTorch的采样器基类

import dinov2.distributed as distributed  # 导入分布式训练相关模块


class EpochSampler(Sampler):
    def __init__(
            self,
            *,
            size: int,  # 总样本数量
            sample_count: int,  # 每次采样的数量
            shuffle: bool = False,  # 是否打乱采样顺序，默认为False
            seed: int = 0,  # 随机种子，默认为0
            start: Optional[int] = None,  # 起始索引，默认为分布式进程的全局排名
            step: Optional[int] = None,  # 采样步长，默认为分布式进程的全局大小
    ):
        """
        初始化采样器
        参数:
            size: 总样本数量
            sample_count: 每次采样的数量
            shuffle: 是否打乱采样顺序，默认为False
            seed: 随机种子，默认为0
            start: 起始索引，默认为分布式进程的全局排名
            step: 采样步长，默认为分布式进程的全局大小
        """
        self._size = size  # 存储总样本数量
        self._sample_count = sample_count  # 存储每次采样的数量
        self._shuffle = shuffle  # 存储是否打乱的标志
        self._seed = seed  # 存储随机种子
        self._start = distributed.get_global_rank() if start is None else start  # 设置起始索引
        self._step = distributed.get_global_size() if step is None else step  # 设置采样步长
        self._epoch = 0  # 初始化轮次为0

    def __iter__(self):
        """
        迭代器方法，生成采样索引
        计算采样次数，根据是否打乱决定采样方式
        如果打乱，则使用随机种子和当前轮次生成随机采样
        否则，按顺序采样
        """
        count = (self._size + self._sample_count - 1) // self._sample_count  # 计算采样次数
        tiled_indices = np.tile(np.arange(self._sample_count), count)  # 重复采样索引数组
        if self._shuffle:  # 如果需要打乱
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch  # 计算随机种子
            rng = np.random.default_rng(seed)  # 创建随机数生成器
            iterable = rng.choice(tiled_indices, self._size, replace=False)  # 随机选择不重复的索引
        else:  # 否则按顺序采样
            iterable = tiled_indices[: self._size]  # 取前_size个索引

        yield from itertools.islice(iterable, self._start, None, self._step)  # 按步长生成迭代器

    def __len__(self):
        """
        返回采样器的长度，即采样次数
        根据起始索引、总样本数和步长计算
        """
        return (self._size - self._start + self._step - 1) // self._step  # 计算采样次数

    def set_epoch(self, epoch):
        """
        设置当前轮次
        参数:
            epoch: 当前轮次编号
        """
        self._epoch = epoch  # 设置当前轮次


def _get_numpy_dtype(size: int) -> Any:

    """根据大小返回合适的NumPy数据类型"""
    return np.int32 if size <= 2 ** 31 else np.int64  # 根据大小返回32位或64位整数


def _get_torch_dtype(size: int) -> Any:

    """根据大小返回合适的PyTorch数据类型"""
    return torch.int32 if size <= 2 ** 31 else torch.int64  # 根据大小返回32位或64位整数


def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


class InfiniteSampler(Sampler):
    def __init__(self, *, sample_count: int, shuffle: bool = False, seed: int = 0, start: Optional[int] = None,
                 step: Optional[int] = None, advance: int = 0):
        """
        初始化函数，用于设置数据集的基本参数
        参数:
            sample_count: 样本总数
            shuffle: 是否打乱样本顺序，默认为False
            seed: 随机种子，默认为0
            start: 起始索引，如果为None则使用分布式全局排名，默认为None
            step: 步长，如果为None则使用分布式全局大小，默认为None
            advance: 额外跳过的样本数，默认为0
        """
        super().__init__()
        self._sample_count = sample_count  # 存储样本总数
        self._seed = seed  # 存储随机种子
        self._shuffle = shuffle  # 存储是否打乱的标志
        self._start = distributed.get_global_rank() if start is None else start  # 设置起始索引
        self._step = distributed.get_global_size() if step is None else step  # 设置步长
        self._advance = advance  # 存储额外跳过的样本数

    def __iter__(self):
        # 根据是否打乱选择迭代器
        if self._shuffle:
            iterator = self._shuffled_iterator()  # 使用打乱顺序的迭代器
        else:
            iterator = self._iterator()  # 使用顺序迭代器

        # 跳过前_advance个样本，然后返回剩余的所有样本
        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        # 断言确保没有打乱顺序
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)  # 创建一个可迭代范围
            yield from itertools.islice(iterable, self._start, None, self._step)  # 按照起始索引和步长生成样本

    def _shuffled_iterator(self):
        # 断言确保打乱顺序
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


# The following function is somewhat equivalent to _new_shuffle_tensor_slice below,
# but avoids a full in-place random permutation generation.
def _shuffle_tensor_slice(
        *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")

    dtype = _get_numpy_dtype(stop)
    result = np.empty(count, dtype=dtype)

    for i in range(count):
        j = torch.randint(0, i + 1, size=(1,), generator=generator).item() if i > 0 else 0

        result[i] = result[j]
        result[j] = tensor[start + i * step].item()

    return result


def _new_shuffle_tensor_slice(
        *, tensor: torch.Tensor, start: int = 0, step: int = 1, generator: torch.Generator
) -> np.ndarray:
    stop = len(tensor)
    count = stop // step
    dtype = torch.int64  # Needed for using randperm result as indices
    count = stop // step
    drop_count = stop - step * count
    if drop_count:
        warnings.warn(f"# of dropped samples: {drop_count}")
    indices = torch.randperm(count, dtype=dtype, generator=generator)
    return tensor[start::step][indices].numpy()


def _make_seed(seed: int, start: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + start + (iter_count << 24)


class ShardedInfiniteSampler(Sampler):
    def __init__(self, *, sample_count: int, shuffle: bool = False, seed: int = 0, start: Optional[int] = None,
                 step: Optional[int] = None, advance: int = 0, use_new_shuffle_tensor_slice: bool = False):
        super().__init__()
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance
        self._iter_count = 0
        self._shuffle_tensor_slice_fn = (
            _new_shuffle_tensor_slice if use_new_shuffle_tensor_slice else _shuffle_tensor_slice
        )

    def __iter__(self):
        iter_count = self._advance // self._sample_count
        if iter_count > 0:
            self._advance -= iter_count * self._sample_count
            self._iter_count += iter_count

        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to be keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator()

        # Always shuffle everything first
        generator.manual_seed(self._seed)
        dtype = _get_torch_dtype(self._sample_count)
        perm = torch.randperm(self._sample_count, dtype=dtype, generator=generator)

        while True:
            # Re-seed on each iteration to allow skipping whole permutations
            seed = _make_seed(self._seed, self._start, self._iter_count)
            generator.manual_seed(seed)

            iterable = self._shuffle_tensor_slice_fn(
                tensor=perm, start=self._start, step=self._step, generator=generator
            )
            yield from iterable
            self._iter_count += 1
