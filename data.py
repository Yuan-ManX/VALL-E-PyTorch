import copy
import logging
import random
from collections import defaultdict
from functools import cache, cached_property
from itertools import groupby, zip_longest
from typing import Any
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import cfg
from sampler import Sampler


# 设置多进程共享策略为 "file_system" 以避免潜在的共享内存问题
torch.multiprocessing.set_sharing_strategy("file_system")


# 获取当前模块的日志记录器
_logger = logging.getLogger(__name__)


def _replace_file_extension(path, suffix):
    """
    替换文件路径的扩展名为指定的后缀。

    参数:
        path (Path): 原始文件路径。
        suffix (str): 新的文件扩展名。

    返回:
        Path: 替换扩展名后的文件路径。
    """
    return (path.parent / path.name.split(".")[0]).with_suffix(suffix)


def _get_quant_path(path):
    """
    获取量化文件的路径。

    参数:
        path (Path): 原始文件路径。

    返回:
        Path: 量化文件的路径（扩展名为 .qnt.pt）。
    """
    return _replace_file_extension(path, ".qnt.pt")


def _load_quants(path) -> Tensor:
    """
    从量化文件中加载量化数据。

    参数:
        path (Path): 量化文件的路径。

    返回:
        Tensor: 量化数据张量，形状为 (t, q)。
    """
    path = _get_quant_path(path)
    return torch.load(path)[0].t()


@cache
def _get_phones(path):
    """
    从音素文件中加载音素列表。

    参数:
        path (Path): 原始文件路径。

    返回:
        List[str]: 音素列表，首尾添加了 <s> 和 </s> 标签。
    """
    # 获取音素文件的路径（扩展名为 .phn.txt）
    path = _replace_file_extension(path, ".phn.txt")
    with open(path, "r", encoding="utf8") as f:
        content = f.read()
    # 将内容拆分为列表，并在首尾添加 <s> 和 </s> 标签
    return ["<s>"] + content.split() + ["</s>"]


def _interleaved_reorder(l, fn):
    """
    对列表中的元素进行交错重排序。

    参数:
        l (List[Any]): 输入列表。
        fn (Callable[[Any], Any]): 用于对元素进行分组的函数。

    返回:
        Generator[Any, None, None]: 交错重排序后的元素生成器。
    """
    # 创建一个默认字典用于存储分组后的元素
    groups = defaultdict(list)
    # 将列表中的每个元素根据函数 fn 的返回值进行分组
    for e in l:
        groups[fn(e)].append(e)
    # 对分组后的键进行排序
    groups = {k: groups[k] for k in sorted(groups)}
    # 对每个分组进行交错遍历
    for interleaved in zip_longest(*groups.values()):
        # 对每个交错组中的元素进行遍历
        for value in interleaved:
            # 如果元素不为空，则生成该元素
            if value is not None:
                yield value


@cache
def _validate(path, min_phones, max_phones):
    """
    验证音素文件是否符合要求。

    参数:
        path (Path): 音素文件的路径。
        min_phones (int): 最小允许的音素数量。
        max_phones (int): 最大允许的音素数量。

    返回:
        bool: 如果音素文件符合要求，则返回 True；否则返回 False。
    """
    # 从音素文件中加载音素列表
    phones = _get_phones(path)
    # 获取唯一的音素集合
    unique_phones = list(set(phones))

    # 验证音素列表是否为空
    if len(unique_phones) == 0:
        return False
    
    # 验证音素列表是否仅包含单个占位符
    if len(unique_phones) == 1 and unique_phones[0] == "_":
        return False
    
    # 验证音素数量是否少于最小允许值
    if len(phones) < min_phones:
        return False
    
    # 验证音素数量是否超过最大允许值
    if len(phones) > max_phones:
        return False
    # 如果所有验证通过，则返回 True
    return True


class VALLEDatset(Dataset):
    """
    VALLEDatset 类实现了一个自定义的数据集类，用于语音合成任务。
    该数据集类支持加载和处理音频文件、文本和量化数据，并提供训练和评估模式下的数据采样方式。

    参数说明:
        paths (List[Path]): 音频文件路径列表。
        phone_symmap (Dict[str, int], 可选): 音素到索引的映射字典。如果未指定，则根据数据自动生成。
        spkr_symmap (Dict[str, int], 可选): 说话人到索引的映射字典。如果未指定，则根据数据自动生成。
        min_phones (int, 可选): 最小允许的音素数量，默认为配置参数中的 min_phones。
        max_phones (int, 可选): 最大允许的音素数量，默认为配置参数中的 max_phones。
        training (bool, 可选): 是否为训练模式，默认为 False。
        extra_paths_by_spkr_name (dict[str, list], 可选): 额外添加的音频文件路径，按说话人名称分类，默认为空字典。
    """
    def __init__(
        self,
        paths,
        phone_symmap=None,
        spkr_symmap=None,
        min_phones=cfg.min_phones,
        max_phones=cfg.max_phones,
        training=False,
        extra_paths_by_spkr_name: dict[str, list] = {},
    ):
        super().__init__()
        # 用于限制数据集长度
        self._head = None
        # 最小允许的音素数量
        self.min_phones = min_phones
        # 最大允许的音素数量
        self.max_phones = max_phones

        # 过滤掉不符合音素数量要求的路径
        self.paths = [
            path for path in paths if _validate(path, self.min_phones, self.max_phones)
        ]

        # 如果未提供说话人符号映射，则自动生成
        self.spkr_symmap = spkr_symmap or self._get_spkr_symmap()
        # 如果未提供音素符号映射，则自动生成
        self.phone_symmap = phone_symmap or self._get_phone_symmap()
        # 设置是否为训练模式
        self.training = training

        # 获取按说话人名称分类的路径列表
        self.paths_by_spkr_name = self._get_paths_by_spkr_name(extra_paths_by_spkr_name)

        # 过滤掉那些说话人只有一个样本的路径
        self.paths = [
            p for p in self.paths if len(self.paths_by_spkr_name[cfg.get_spkr(p)]) > 1
        ]

        # 如果没有有效的路径并且是训练模式，则抛出异常
        if len(self.paths) == 0 and training:
            raise ValueError("No valid path is found for training.")

        # 如果是训练模式，则使用采样器进行采样
        if training:
            self.sampler = Sampler(self.paths, [cfg.get_spkr])
        else:
            self.sampler = None

    def _get_paths_by_spkr_name(self, extra_paths_by_spkr_name: dict[str, list]):
        """
        获取按说话人名称分类的路径列表。

        参数:
            extra_paths_by_spkr_name (dict[str, list]): 额外添加的音频文件路径，按说话人名称分类。

        返回:
            dict[str, list]: 按说话人名称分类的路径列表。
        """
        ret = defaultdict(list)
        for path in self.paths:
            if _get_quant_path(path).exists():
                ret[cfg.get_spkr(path)].append(path)
        for k, v in extra_paths_by_spkr_name.items():
            ret[k].extend(v)
        return {**ret}

    @cached_property
    def phones(self):
        """
        获取所有独特的音素列表。

        Returns:
            List[str]: 所有独特的音素列表。
        """
        return sorted(set().union(*[_get_phones(path) for path in self.paths]))

    def _get_phone_symmap(self):
        """
        生成音素到索引的映射字典。

        Returns:
            dict[str, int]: 音素到索引的映射字典。
        """
        # 注意我们从1开始编号，以便可以安全地填充0。
        return {s: i for i, s in enumerate(self.phones, 1)}

    @cached_property
    def spkrs(self):
        """
        获取所有独特的说话人列表。

        Returns:
            List[str]: 所有独特的说话人列表。
        """
        return sorted({cfg.get_spkr(path) for path in self.paths})

    def _get_spkr_symmap(self):
        """
        生成说话人到索引的映射字典。

        Returns:
            dict[str, int]: 说话人到索引的映射字典。
        """
        return {s: i for i, s in enumerate(self.spkrs)}

    def sample_prompts(self, spkr_name, ignore):
        """
        对指定说话人进行提示样本采样。

        参数:
            spkr_name (str): 说话人名称。
            ignore (Path): 要忽略的路径。

        Returns:
            Tensor: 采样后的提示样本张量。
        """
        prom_list = []

        choices = set(self.paths_by_spkr_name[spkr_name]) - {ignore}
        choices = [*choices]

        if len(choices) == 0:
            raise ValueError(
                f"Failed to find another different utterance for {spkr_name}."
            )

        for _ in range(cfg.max_prompts):
            path = random.choice(choices)
            prom_list.append(_load_quants(path))
            if random.random() > cfg.p_additional_prompt:
                break

        prom = torch.cat(prom_list)

        return prom

    def __getitem__(self, index):
        """
        获取指定索引的数据样本。

        参数:
            index (int): 数据样本的索引。

        Returns:
            dict: 数据样本，包括路径、说话人名称、文本、提示样本和响应。
        """
        if self.training:
            assert self.sampler is not None
            path = self.sampler.sample()
        else:
            path = self.paths[index]

        spkr_name = cfg.get_spkr(path)
        text = torch.tensor([*map(self.phone_symmap.get, _get_phones(path))])
        proms = self.sample_prompts(spkr_name, ignore=path)
        resps = _load_quants(path)
        resp = resps[..., 0]

        return dict(
            path=path,
            spkr_name=spkr_name,
            text=text,
            proms=proms,
            resps=resps,
            resp=resp,
        )

    def head_(self, n):
        """
        设置数据集的最大长度。

        参数:
            n (int): 最大长度。
        """
        self._head = n

    def training_(self, value):
        """
        设置是否为训练模式。

        参数:
            value (bool): 是否为训练模式。
        """
        self.training = value

    def interleaved_reorder_(self, fn):
        """
        对数据集路径进行交错重排序。

        参数:
            fn (Callable[[Path], Any]): 用于对路径进行分组的函数。
        """
        self.paths = [*_interleaved_reorder(self.paths, fn)]

    def __len__(self):
        """
        获取数据集的长度。

        Returns:
            int: 数据集的长度。
        """
        return min(len(self.paths), self._head or len(self.paths))


# 数据批处理函数
def collate_fn(samples: list[dict]):
    """
    对数据样本列表进行批处理，将相同键的值收集到列表中。

    参数:
        samples (List[Dict[str, Any]]): 数据样本列表，每个样本是一个字典。

    返回:
        Dict[str, Any]: 批处理后的数据字典，键为样本的键，值为对应值的列表。
    """
    # 使用字典推导式，将每个样本的相同键的值收集到列表中
    batch: dict[str, Any] = {k: [s[k] for s in samples] for k in samples[0]}
    return batch


# 随机种子设置函数，用于DataLoader的worker
def _seed_worker(worker_id):
    """
    为DataLoader的每个worker设置随机种子，确保数据加载的可重复性。

    参数:
        worker_id (int): worker的ID。
    """
    # 计算worker的种子，确保不同worker使用不同的种子
    worker_seed = torch.initial_seed() % 2**32
    # 设置numpy和python内置的随机数生成器的种子
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 创建DataLoader的函数
def _create_dataloader(dataset, training):
    """
    根据训练或评估模式创建DataLoader。

    参数:
        dataset (Dataset): 要加载的数据集。
        training (bool): 是否为训练模式。

    返回:
        DataLoader: 创建的DataLoader对象。
    """
    # 创建DataLoader
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size if training else cfg.eval_batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=cfg.nj,
        collate_fn=collate_fn,
        persistent_workers=True,
        worker_init_fn=_seed_worker,
    )


# 加载训练和验证路径的函数
def _load_train_val_paths():
    """
    加载训练和验证数据集的路径。

    返回:
        Tuple[List[Path], List[Path]]: 训练路径列表和验证路径列表。
    """
    paths = []
    train_paths = []
    val_paths = []

    # 遍历所有数据目录，查找所有 .qnt.pt 文件
    for data_dir in cfg.data_dirs:
        paths.extend(tqdm(data_dir.rglob("*.qnt.pt")))

    if len(paths) == 0:
        raise RuntimeError(f"Failed to find any .qnt.pt file in {cfg.data_dirs}.")
    
    # 按说话人排序所有路径
    pairs = sorted([(cfg.get_spkr(p), p) for p in paths])
    del paths

    # 按说话人分组，并打乱每个组内的路径
    for _, group in groupby(pairs, lambda pair: pair[0]):
        paths = sorted([p for _, p in group])
        # 设置随机种子以确保可重复性
        random.seed(0)
        # 打乱路径顺序
        random.shuffle(paths)
        # 计算训练样本数量（95%）
        n = round(len(paths) * 0.95)
        # 添加到训练路径列表
        train_paths.extend(paths[:n])
        # 添加到验证路径列表
        val_paths.extend(paths[n:])

    # 对训练和验证路径进行排序
    train_paths, val_paths = map(sorted, [train_paths, val_paths])

    return train_paths, val_paths


@cfg.diskcache()
def create_datasets():
    """
    创建训练和验证数据集。

    返回:
        Tuple[Dataset, Dataset]: 训练数据集和验证数据集。
    """
    # 加载训练和验证路径
    train_paths, val_paths = _load_train_val_paths()

    # 创建训练数据集
    train_dataset = VALLEDatset(
        train_paths,
        training=True,
    )

    # 创建验证数据集
    val_dataset = VALLEDatset(
        val_paths,
        train_dataset.phone_symmap, # 使用训练数据集的音素符号映射
        train_dataset.spkr_symmap, # 使用训练数据集的说话人符号映射
        extra_paths_by_spkr_name=train_dataset.paths_by_spkr_name, # 使用训练数据集的额外路径
    )

    # 对验证数据集进行交错重排序
    val_dataset.interleaved_reorder_(cfg.get_spkr)
    # 设置验证数据集的最大样本数量
    val_dataset.head_(cfg.max_num_val)

    return train_dataset, val_dataset


def create_train_val_dataloader():
    """
    创建训练、验证和子训练 DataLoader。

    返回:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练 DataLoader、子训练 DataLoader 和验证 DataLoader。
    """
    # 创建训练和验证数据集
    train_dataset, val_dataset = create_datasets()

    # 创建训练 DataLoader
    train_dl = _create_dataloader(train_dataset, training=True)
    # 创建验证 DataLoader
    val_dl = _create_dataloader(val_dataset, training=False)

    # 记录音素符号映射和说话人符号映射
    _logger.info(str(train_dataset.phone_symmap))
    _logger.info(str(train_dataset.spkr_symmap))

    # 记录训练和验证样本数量
    _logger.info(f"#samples (train): {len(train_dataset)}.")
    _logger.info(f"#samples (val): {len(val_dataset)}.")

    # 创建子训练数据集（用于验证）
    subtrain_dataset = copy.deepcopy(train_dataset)
    # 对子训练数据集进行交错重排序
    subtrain_dataset.interleaved_reorder_(cfg.get_spkr)
    # 设置子训练数据集的最大样本数量
    subtrain_dataset.head_(cfg.max_num_val)
    # 设置子训练数据集为非训练模式
    subtrain_dataset.training_(False)
    # 创建子训练 DataLoader
    subtrain_dl = _create_dataloader(subtrain_dataset, training=False)
    assert isinstance(subtrain_dl.dataset, VALLEDatset)

    return train_dl, subtrain_dl, val_dl


if __name__ == "__main__":

    # 创建训练、验证和子训练 DataLoader
    train_dl, subtrain_dl, val_dl = create_train_val_dataloader()
    # 获取训练数据集的第一个样本
    sample = train_dl.dataset[0]
    print(sample)
