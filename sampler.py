import random


class Sampler:
    """
    Sampler 类实现了一个多级随机采样器。
    该类通过多级键函数将数据集分层，并支持在每个层级内进行随机采样。
    适用于需要根据多个属性进行分层采样的场景，例如按说话人和音素进行分层采样。

    参数说明:
        l (List[Any]):): 要采样的数据集列表。
        key_fns (List[Callable[[Any], Any]]): 用于分层的键函数列表。
            第一个键函数用于最外层分组，第二个键函数用于次外层分组，依此类推。
    """
    def __init__(self, l, key_fns):
        """
        初始化 Sampler 对象。

        参数:
            l (List[Any]): 要采样的数据集列表。
            key_fns (List[Callable[[Any], Any]]): 用于分层的键函数列表。
        """
        # 构建分层树结构
        self.tree = self._build(l, key_fns)

    def _build(self, l, key_fns) -> dict[dict, list]:
        """
        递归地构建分层树结构。

        参数:
            l (List[Any]): 当前层级的数据集列表。
            key_fns (List[Callable[[Any], Any]]): 剩余的键函数列表。

        返回:
            Dict[Any, Any]: 分层树结构。
        """
        if not key_fns:
            # 如果没有更多的键函数，则返回当前列表
            return l

        tree = {}

        # 获取当前的键函数，并更新键函数列表
        key_fn, *key_fns = key_fns

        for x in l:
            # 计算当前元素的键
            k = key_fn(x)

            if k in tree:
                # 如果键已存在，则将元素添加到对应的列表中
                tree[k].append(x)
            else:
                # 如果键不存在，则创建一个新的列表
                tree[k] = [x]

        # 递归地为每个子列表构建树结构
        for k in tree:
            tree[k] = self._build(tree[k], key_fns)

        return tree

    def _sample(self, tree: dict | list):
        """
        递归地进行随机采样。

        参数:
            tree (Dict[Any, Any] | List[Any]): 当前层级的树结构或列表。

        返回:
            Any: 采样的元素。
        """
        if isinstance(tree, list):
            # 如果当前层级是列表，则进行随机选择
            ret = random.choice(tree)
        else:
            # 如果当前层级是字典，则随机选择一个键，并递归采样
            key = random.choice([*tree.keys()])
            ret = self._sample(tree[key])
        return ret

    def sample(self):
        """
        执行一次随机采样。

        返回:
            Any: 采样的元素。
        """
        return self._sample(self.tree)
