import torch
from torch import Tensor

from base import Base


class NAR(Base):
    """
    NAR 类继承自 Base 类，实现了一个非自回归（Non-Autoregressive）模型。
    该模型在生成响应时，可以同时生成多个标记，而不需要逐个生成。
    """
    @property
    def n_resp_levels(self):
        """
        响应级别的数量。

        Returns:
            int: 响应级别的数量，默认为 7。
        """
        return 7

    @property
    def casual(self):
        """
        是否使用因果掩码。

        Returns:
            bool: 如果使用因果掩码，则返回 True；否则返回 False。
        """
        return False

    @property
    def use_stop_token(self):
        """
        是否使用停止标记。

        Returns:
            bool: 如果使用停止标记，则返回 True；否则返回 False。
        """
        return False

    @property
    def norm_type(self):
        """
        归一化类型。

        Returns:
            str: 归一化类型，默认为 'adaln'（自适应层归一化）。
        """
        return "adaln"

    @property
    def resp_loss_only(self):
        """
        是否仅计算响应损失。

        Returns:
            bool: 如果仅计算响应损失，则返回 True；否则返回 False。
        """
        return True

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        sampling_temperature: float = 0.2,
    ):
        """
        前向传播方法，执行非自回归生成。

        参数:
            text_list (List[Tensor]): 输入文本列表，形状为 [t] * b。
            proms_list (List[Tensor]): 输入提示列表，形状为 [t' l] * b，l=8。
            resps_list (List[Tensor]): 输入响应列表，形状为 [t'' l] * b，l=1 或 8，1 表示测试，8 表示训练。
            sampling_temperature (float, 可选): 采样温度，默认为 0.2。

        返回:
            List[Tensor]: 测试时返回 [t'' l]，l=8；训练时返回空列表。
        """
        # 获取响应列表中每个响应的级别数量
        n_levels_set = {r.shape[-1] for r in resps_list}

        # 如果级别数量不一致，则抛出异常
        if len(n_levels_set) > 1:
            raise ValueError(f"Please give only one level, got {n_levels_set}.")

        # 获取当前的级别数量
        n_levels = next(iter(n_levels_set))

        device = text_list[0].device

        if n_levels == self.n_resp_levels + 1:
            assert resps_list is not None

            # 随机生成量化级别
            quant_levels = torch.randint(0, self.n_resp_levels, (len(resps_list),))

            # 获取之前的响应列表
            prev_list = [o[..., : l + 1] for o, l in zip(resps_list, quant_levels)]
            # 获取目标响应列表
            targ_list = [o[..., l + 1] for o, l in zip(resps_list, quant_levels)]

            quant_levels = quant_levels.to(device=device)

            # 调用 Base 类的 forward 方法
            _ = super().forward(
                text_list,
                proms_list,
                prev_list,
                targ_list,
                return_all_resp=True, # 返回所有响应
                shift_targ_list=False, # 不移动目标列表
                quant_levels=quant_levels, # 指定量化级别
            )

            # 在训练时，返回空列表
            prev_list = []
        else:
            # 初始化之前的响应列表
            prev_list = resps_list

            while True:
                # 获取当前级别
                level = prev_list[0].shape[-1] - 1

                if level >= self.n_resp_levels:
                    # 如果当前级别达到最大级别，则退出循环
                    break
                
                # 生成量化级别张量
                quant_levels = torch.full((len(text_list),), level, device=device)

                # 调用 Base 类的 forward 方法
                resp_list = super().forward(
                    text_list,
                    proms_list,
                    prev_list,
                    return_all_resp=True, # 返回所有响应
                    shift_targ_list=False, # 不移动目标列表
                    quant_levels=quant_levels, # 指定量化级别
                    sampling_temperature=sampling_temperature, # 指定采样温度
                )

                # 更新之前的响应列表
                prev_list = [
                    torch.cat([rs, r.unsqueeze(-1)], dim=-1)
                    for rs, r in zip(prev_list, resp_list)
                ]
        # 返回最终的响应列表
        return prev_list

