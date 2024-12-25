import torch
from einops import rearrange
from torch import Tensor
from tqdm import trange

from base import Base


class AR(Base):
    """
    AR 类继承自 Base 类，实现了一个自回归（Autoregressive）模型。
    该模型在生成响应时，逐个生成每个标记，并且可以使用停止标记来终止生成过程。
    """
    @property
    def n_resp_levels(self):
        """
        响应级别的数量。

        Returns:
            int: 响应级别的数量，默认为 1。
        """
        return 1

    @property
    def casual(self):
        """
        是否使用因果掩码。

        Returns:
            bool: 如果使用因果掩码，则返回 True；否则返回 False。
        """
        return True

    @property
    def use_stop_token(self):
        """
        是否使用停止标记。

        Returns:
            bool: 如果使用停止标记，则返回 True；否则返回 False。
        """
        return True

    @property
    def norm_type(self):
        """
        归一化类型。

        Returns:
            str: 归一化类型，默认为 'ln'（层归一化）。
        """
        return "ln"

    @property
    def resp_loss_only(self):
        """
        是否仅计算响应损失。

        Returns:
            bool: 如果仅计算响应损失，则返回 True；否则返回 False。
        """
        return False

    def _prune(self, l: Tensor):
        """
        修剪响应序列，移除停止标记及其之后的所有标记。

        参数:
            l (Tensor): 响应序列张量。

        Returns:
            Tensor: 修剪后的响应序列张量。
        """
        # 查找停止标记的索引
        indices = (l == self.stop_token).nonzero()
        if len(indices) == 0:
            # 如果没有找到停止标记，则返回原始序列
            return l
        # 返回停止标记之前的所有标记
        return l[: indices.min().item()]

    @staticmethod
    def _unsqueeze_list(x_list, axis=-1):
        """
        对张量列表中的每个张量增加一个维度。

        参数:
            x_list (List[Tensor]): 输入张量列表。
            axis (int, 可选): 要增加的维度，默认为 -1。

        Returns:
            List[Tensor]: 增加维度后的张量列表。
        """
        return [x.unsqueeze(dim=axis) for x in x_list]

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resp_list: list[Tensor] | None = None,
        max_steps: int = 1000,
        sampling_temperature: float = 1.0,
    ):
        """
        前向传播方法，执行自回归生成或计算损失。

        参数:
            text_list (List[Tensor]): 输入文本列表，形状为 [t] * b。
            proms_list (List[Tensor]): 输入提示列表，形状为 [t' l] * b，l 表示量化级别。
            resp_list (List[Tensor], 可选): 输入响应列表，形状为 [t'' l] * b，l 表示量化级别。如果提供，则计算损失；否则，执行生成。
            max_steps (int, 可选): 最大生成步数，默认为 1000。
            sampling_temperature (float, 可选): 采样温度，默认为 1.0。

        Returns:
            Union[Tensor, List[Tensor]]: 如果提供 resp_list，则返回损失张量；否则，返回生成结果列表。
        """
        if resp_list is not None:
            return super().forward(
                text_list,
                proms_list,
                self._unsqueeze_list(resp_list),  # 对响应列表中的每个张量增加一个维度
                targ_list=resp_list,  # 设置目标列表为响应列表
                quant_levels=None,  # 不指定量化级别
                shift_targ_list=True,  # 在自回归模式下移动目标列表
                return_all_resp=False,  # 不返回所有响应
            )
        else:
            return self._generate(
                text_list,
                proms_list,
                max_steps,
                sampling_temperature,
            )

    def _generate(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        max_steps: int,
        sampling_temperature: float,
    ):
        """
        生成响应序列。

        参数:
            text_list (List[Tensor]): 输入文本列表，形状为 [t] * b。
            proms_list (List[Tensor]): 输入提示列表，形状为 [t' l] * b，l 表示量化级别。
            max_steps (int): 最大生成步数。
            sampling_temperature (float): 采样温度。

        Returns:
            List[Tensor]: 生成的结果列表。
        """
        device = text_list[0].device
        # 初始化响应列表，每个响应为空张量
        resp_list: list[Tensor] = [
            torch.zeros(0, device=device).long() for _ in text_list
        ]
        # 初始化停止标志
        stopped = torch.zeros(len(text_list), device=device).bool()
        # 迭代最大生成步数
        for _ in trange(max_steps):
            # 执行前向传播
            r = super().forward( 
                text_list,
                proms_list, 
                self._unsqueeze_list(resp_list),  # 对响应列表中的每个张量增加一个维度
                sampling_temperature=sampling_temperature,  # 设置采样温度
            )
            # 更新停止标志
            stopped |= r == self.stop_token
            # 遍历生成的每个响应
            for i, ri in enumerate(r):
                # 将新生成的标记添加到响应列表中
                resp_list[i] = torch.cat([resp_list[i], ri[None]])
            if stopped.all().item():
                # 如果所有样本都已停止，则退出循环
                break
        
        # 修剪响应序列，移除停止标记及其之后的所有标记
        pruned = [self._prune(r) for r in resp_list]
        # 返回修剪后的响应列表
        return pruned


def example_usage():
    """
    示例函数，展示如何使用 AR 模型进行训练和推理。
    该函数加载量化数据、初始化模型、准备数据、执行训练，并生成音频文件。
    """
    from functools import partial

    import soundfile
    from einops import repeat

    device = "cuda"

    # 加载量化数据
    qnt = torch.load("data/test/test.qnt.pt")[0, 0].to(device)
    # 量化级别数量
    num_qnts = 1024
    
    # 初始化 AR 模型
    model = AR(num_qnts).to(device)

    # 准备文本输入列表
    text_list = [
        torch.tensor([1, 2, 3], device=device),  # 第一个文本序列
        torch.tensor([2, 3], device=device),  # 第二个文本序列
    ]

    # 定义一个偏函数，用于重复张量以匹配提示的维度
    x8 = partial(repeat, pattern="t -> t l", l=8)
    # 准备提示输入列表
    proms_list = [
        x8(torch.tensor([1, 2, 3], device=device)),  # 第一个提示序列
        x8(torch.tensor([2, 3], device=device)),  # 第二个提示序列
    ]  

    # 准备响应输入列表
    resp_list = [
        torch.tensor([1, 2, 3], device=device),  # 第一个响应序列
        qnt.to(device),  # 第二个响应序列（量化数据）
    ]
    
    # 使用 AR 模型进行推理，生成输出
    out = model(text_list, proms_list, max_steps=200)

    print(out)

    # 定义优化器
    # 使用 Adam 优化器，学习率为 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for i in range(100):
        optimizer.zero_grad()
        # 前向传播，计算损失
        _ = model(text_list, proms_list, resp_list)

        # 获取损失
        losses = model.loss
        # 计算总损失并反向传播
        sum(losses.values()).backward()
        # 更新模型参数
        optimizer.step()

        if i % 20 == 0:
            # 每20次迭代打印一次损失
            print(f"iter={i}, {losses}.")

    # 再次使用 AR 模型进行推理，生成输出
    out = model(text_list, proms_list, max_steps=200)

    print(qnt) # 打印原始量化数据
    print(out) # 打印最终生成结果
    
    # 解码生成结果并保存为音频文件
    from emb.qnt import decode

    # 重塑张量形状
    codes = rearrange(out[1], "t -> 1 1 t")
    # 解码生成结果为音频波形和采样率
    wavs, sr = decode(codes)
    soundfile.write("data/test/test.ar.recon.wav", wavs.cpu()[0, 0], sr)


if __name__ == "__main__":

    example_usage()
