import math
from functools import partial
from typing import Literal, overload

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


def _create_mask(l, device):
    """1 is valid region and 0 is invalid."""
    """
    创建掩码张量，其中1表示有效区域，0表示无效区域。

    参数:
        l (List[int]): 每个样本的长度列表。
        device (torch.device): 张量所在的设备。

    返回:
        torch.Tensor: 掩码张量，形状为 (batch_size, sequence_length)。
    """
    # 生成一个从0到最大长度的序列张量，形状为 (1, t)
    seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
    # 将长度列表转换为张量，并增加一个维度，形状为 (batch_size, 1)
    stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
    # 生成掩码张量，条件为序列小于长度，形状为 (batch_size, sequence_length)
    return (seq < stop).float()  # (b t)


def list_to_tensor(x_list: list[Tensor], pattern="t b c -> b t c"):
    """
    Args:
        x_list: [(t d)]
    Returns:
        x: (? ? ?)
        m: (? ? ?), same as x
    """
    """
    将张量列表转换为批处理张量，并创建相应的掩码。

    参数:
        x_list (List[torch.Tensor]): 输入张量列表，每个张量的形状为 (t, d)。
        pattern (str, 可选): 重塑张量的模式，默认为 "t b c -> b t c"。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 批处理后的张量和对应的掩码，形状由 pattern 决定。
    """
    # 获取每个张量的长度列表
    l = list(map(len, x_list))
    # 使用 pad_sequence 对张量列表进行填充，使其长度一致，并重塑为 (batch_size, max_length, dim)
    x = rearrange(pad_sequence(x_list), pattern)
    # 创建掩码张量，1 表示有效区域，0 表示无效区域
    m = _create_mask(l, x_list[0].device)
    # 将掩码张量转置并增加一个维度，形状为 (t, b, 1)
    m = m.t().unsqueeze(-1)  # (t b 1)
    # 根据指定的模式重塑掩码张量
    m = rearrange(m, pattern)
    # 将掩码张量移动到与输入张量相同的设备
    m = m.to(x)
    return x, m


class SinusodialEmbedding(nn.Module):
    """
    SinusoidalEmbedding 类实现了一个正弦位置编码模块。
    该模块根据输入序列的位置生成正弦和余弦编码，用于在模型中引入位置信息。

    参数说明:
        d_model (int): 模型的维度大小。
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 计算指数值，用于生成正弦和余弦编码
        exponent = torch.arange(self.d_half, dtype=torch.float32)
        exponent = exponent / self.d_half
        # 生成 omega，公式为 omega = exp(-log(1e4) * exponent)
        omega = torch.exp(-math.log(1e4) * exponent)
        self.omega: torch.Tensor
        # 注册 omega 张量为缓冲区，不作为模型参数保存
        self.register_buffer("omega", omega, persistent=False)

    @property
    def d_half(self):
        """
        获取模型维度的一半。

        返回:
            int: 模型维度的一半。
        """
        assert self.d_model % 2 == 0, "Only support even d_model."
        return self.d_model // 2

    def forward(self, x):
        """
        前向传播方法，生成位置编码。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 位置编码张量，形状为 (..., d_model)。
        """
        omega = self.omega

        # 确保 omega 的维度与 x 的维度一致
        while omega.dim() <= x.dim():
            # 在最前面增加一个维度
            omega = omega.unsqueeze(0)  # (... d)

        # 在最后增加一个维度，形状为 (..., 1)
        x = x.unsqueeze(-1)  # (... 1)
        # 将 omega 与 x 相乘
        x = omega * x
        # 拼接正弦和余弦编码
        x = torch.cat([x.sin(), x.cos()], dim=-1)

        return x

    def get_pe(self, n: int):
        """
        获取指定长度的位置编码。

        参数:
            n (int): 序列长度。

        返回:
            Tensor: 位置编码张量，形状为 (n, d_model)。
        """
        device = self.omega.device
        return self.forward(torch.arange(n, device=device))

    def add_pe(self, x):
        """
        将位置编码添加到输入张量中。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, d_model)。

        返回:
            Tensor: 添加位置编码后的张量，形状为 (batch_size, sequence_length, d_model)。
        """
        # 获取位置编码，形状为 (sequence_length, d_model)
        e = self.get_pe(x.shape[1])  # t d
        # 增加一个维度，形状为 (1, sequence_length, d_model)
        e = e[None]  # b t d
        # 添加位置编码
        x = x + e
        return x


class Attention(nn.Module):
    """
    Attention 类实现了一个多头自注意力机制（Multi-Head Self-Attention）。
    该机制广泛应用于 Transformer 模型中，用于捕捉输入序列中不同位置之间的关系。
    支持因果掩码（casual mask），用于防止模型在预测时看到未来的信息。

    参数说明:
        d_model (int): 模型的维度大小。
        n_heads (int): 注意力头的数量。
        casual (bool): 是否使用因果掩码。
    """
    def __init__(self, d_model, n_heads, casual):
        super().__init__()
        assert d_model % n_heads == 0
        # 每个注意力头的维度
        dim_head = d_model // n_heads
        # 是否使用因果掩码
        self.casual = casual
        # 注意力头的数量
        self.n_heads = n_heads
        # 缩放因子，用于缩放注意力得分
        self.scale = dim_head**-0.5
        # 线性层，用于生成查询（q）、键（k）和值（v）
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        # 线性层，用于输出
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        前向传播方法，执行多头自注意力机制。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, d_model)。
            m (Tensor): 掩码张量，形状为 (batch_size, sequence_length, d_model)，1 表示数据，0 表示填充。

        返回:
            Tensor: 输出张量，形状为 (batch_size, sequence_length, d_model)。
        """
        # 获取注意力头的数量
        h = self.n_heads

        # 将输入张量线性变换为查询（q）、键（k）和值（v）
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # 重塑为多头格式，形状为 (batch_size, sequence_length, n_heads, dim_head)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=h), (q, k, v))

        # 计算注意力得分，形状为 (batch_size, sequence_length, sequence_length, n_heads)
        e = einsum("b i h d, b j h d -> b i j h", q, k)
        # 缩放注意力得分
        e = e * self.scale

        # 生成掩码，形状为 (batch_size, sequence_length, sequence_length, 1)
        kpm = m.unsqueeze(1) * m.unsqueeze(2)  # b i j 1

        if self.casual:
            # 使用因果掩码，形状为 (batch_size, sequence_length, sequence_length, 1)
            kpm = kpm.squeeze(-1).tril().unsqueeze(-1)  # b i j 1

        # 应用掩码，将填充位置填充为负无穷
        e = e.masked_fill(kpm == 0, -torch.finfo(e.dtype).max)
        # 对注意力得分进行 softmax 归一化，形状为 (batch_size, sequence_length, sequence_length, n_heads)
        a = e.softmax(dim=2)  # Normalize on j, i.e. key

        # 计算加权求和，形状为 (batch_size, sequence_length, n_heads, dim_head)
        o = einsum("b i j h, b j h d -> b i h d", a, v)
        # 重塑回 (batch_size, sequence_length, d_model)
        o = o.flatten(-2)
        # 通过线性层输出
        o = self.to_out(o)  # b t c

        # 应用掩码
        o = o * m

        return o


class AdaLN(nn.Module):
    """
    AdaLN 类实现了一个自适应层归一化（Adaptive Layer Normalization）模块。
    该模块根据输入的条件标签动态调整层归一化的参数（gamma 和 beta）。
    适用于需要根据不同条件进行动态调整的场景，如语音合成中的说话人条件。

    参数说明:
        d_model (int): 模型的维度大小。
        n_levels (int): 条件标签的级别数量。
        eps (float, 可选): 层归一化的 epsilon值，默认为1e-5。
        k (float, 可选): AdaLN 中的缩放因子，默认为0.1。
        c (float, 可选): AdaLN 中的常数因子，默认为2。
    """
    def __init__(self, d_model, n_levels, eps=1e-5, k=0.1, c=2):
        super().__init__()
        # 层归一化的 epsilon值
        self.eps = eps
        # 嵌入层，用于生成 gamma 和 beta
        self.emb = nn.Embedding(n_levels, d_model * 2)
        # 缩放因子
        self.k = k
        # 常数因子
        self.c = c
        # 初始化嵌入层的权重为零
        nn.init.zeros_(self.emb.weight)

    def forward(self, x, l):
        """
        前向传播方法，执行自适应层归一化。

        参数:
            x (Tensor): 输入张量。
            l (Tensor): 条件标签张量。

        返回:
            Tensor: 自适应层归一化后的输出张量。
        """
        # 从嵌入层生成 gamma 和 beta
        logγ, β = self.emb(l).unsqueeze(1).chunk(2, dim=-1)
        # 对输入张量进行层归一化
        h = F.layer_norm(x, x.shape[-1:], eps=self.eps)

        # 使用 AdaLN 中的调整公式对 h 进行调整
        h = self.c * (1 - (self.k * h).detach()) * h

        # 应用 gamma 和 beta 进行缩放和平移
        y = logγ.exp() * h + β

        return y


class PrenormResidual(nn.Module):
    """
    PrenormResidual 类实现了一个预归一化残差连接模块。
    该模块在残差连接之前对输入进行归一化处理，支持多种归一化类型，如层归一化（LayerNorm）和自适应层归一化（AdaLN）。

    参数说明:
        block (nn.Module): 要执行的块（通常是神经网络层）。
        d_model (int): 模型的维度大小。
        p_dropout (float): Dropout 层的失活概率。
        requires_mask (bool, 可选): 是否需要掩码，默认为 False。
        norm_type (str, 可选): 归一化类型，'ln' 表示层归一化，'adaln' 表示自适应层归一化，默认为 'ln'。
        n_levels (int, 可选): 条件标签的级别数量，仅在 norm_type 为 'adaln' 时需要。
    """
    def __init__(
        self,
        block,
        d_model,
        p_dropout,
        requires_mask=False,
        norm_type="ln",
        n_levels: int | None = None,
    ):
        super().__init__()
        # 要执行的块
        self.block = block
        # 是否需要掩码
        self.requires_mask = requires_mask
        # 归一化类型
        self.norm_type = norm_type
        if norm_type == "ln":
            # 层归一化
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "adaln":
            assert n_levels is not None
            # 自适应层归一化
            self.norm = AdaLN(d_model, n_levels)
        else:
            raise NotImplementedError(norm_type)
        # Dropout 层
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, m, l):
        """
        前向传播方法，执行预归一化残差连接。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, d_model)。
            m (Tensor): 掩码张量，形状为 (batch_size, sequence_length, 1)，1 表示有价值的数据，0 表示填充。
            l (Tensor): 条件标签张量，仅在 norm_type 为 'adaln' 时需要。

        返回:
            Tensor: 输出张量，形状为 (batch_size, sequence_length, d_model)。
        """
        # 如果归一化类型为 'adaln'，则设置 nopts 参数
        nopts = {"l": l} if self.norm_type == "adaln" else {}
        # 如果需要掩码，则设置 bopts 参数
        bopts = {"m": m} if self.requires_mask else {}
        # 执行残差连接：输入 + Dropout(块(归一化后的输入 * 掩码))
        x = x + self.dropout(self.block(self.norm(x, **nopts) * m, **bopts))
        # 返回应用掩码后的输出
        return x * m


class Block(nn.Sequential):
    """
    Block 类实现了一个 Transformer 块，包含多头自注意力机制和前馈神经网络（FFN）。
    该块使用预归一化残差连接（Prenorm Residual Connection）来稳定训练过程。
    支持因果掩码（casual mask）和多种归一化类型，如层归一化（LayerNorm）和自适应层归一化（AdaLN）。

    参数说明:
        d_model (int): 模型的维度大小。
        n_heads (int): 注意力头的数量。
        p_dropout (float): Dropout 层的失活概率。
        casual (bool): 是否使用因果掩码。
        norm_type (str): 归一化类型，'ln' 表示层归一化，'adaln' 表示自适应层归一化。
        n_levels (int): 条件标签的级别数量，仅在 norm_type 为 'adaln' 时需要。
    """
    def __init__(self, d_model, n_heads, p_dropout, casual, norm_type, n_levels):
        super().__init__()

        # 创建多头自注意力机制的预归一化残差连接块
        self.attn = PrenormResidual(
            Attention(d_model, n_heads, casual),  # 多头自注意力机制
            d_model=d_model,  # 模型维度
            p_dropout=p_dropout,  # Dropout 失活概率
            requires_mask=True,  # 需要掩码
            norm_type=norm_type,  # 归一化类型
            n_levels=n_levels,  # 条件标签级别数量
        )

        # 创建前馈神经网络的预归一化残差连接块
        self.ffn = PrenormResidual(
            nn.Sequential(  # 前馈神经网络
                nn.Linear(d_model, d_model * 4),  # 线性层，将维度扩展4倍
                nn.GELU(),  # 使用 GELU 激活函数
                nn.Dropout(p_dropout),  # Dropout 层
                nn.Linear(d_model * 4, d_model),  # 线性层，恢复原始维度
            ),
            # 模型维度
            d_model=d_model,
            # Dropout 
            p_dropout=p_dropout,
            # 归一化类型
            norm_type=norm_type,
            # 条件标签级别数量
            n_levels=n_levels,
        )

    def forward(self, x, m, l):
        """
        前向传播方法，执行 Transformer 块的前向计算。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, sequence_length, d_model)。
            m (Tensor): 掩码张量，形状为 (batch_size, sequence_length, 1)，1 表示有价值的数据，0 表示填充。
            l (Tensor): 条件标签张量，形状为 (batch_size, )。

        返回:
            Tensor: 输出张量，形状为 (batch_size, sequence_length, d_model)。
        """
        # 假设内存不足
        poor_in_vram = True

        if x.requires_grad and poor_in_vram:
            # 如果输入需要梯度且内存不足，则使用 checkpoint 进行梯度检查点
            x = checkpoint(self.attn, x, m, l)
        else:
            # 否则，正常执行多头自注意力机制
            x = self.attn(x, m, l)
        # 执行前馈神经网络
        x = self.ffn(x, m, l)
        return x


class Embedding(nn.Embedding):
    """
    Embedding 类扩展了 PyTorch 的 Embedding 模块，支持对输入列表进行嵌入。
    该类将输入列表中的每个张量分别进行嵌入，然后返回嵌入后的列表。

    参数说明:
        num_embeddings (int): 嵌入字典的大小。
        embedding_dim (int): 每个嵌入向量的维度。
    """
    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        """
        前向传播方法，对输入列表进行嵌入。

        参数:
            x_list (List[Tensor]): 输入张量列表。

        返回:
            List[Tensor]: 嵌入后的张量列表。
        """
        if len(x_list) == 0:
            # 如果输入列表为空，则返回空列表
            return []
        
        # 将输入列表中的所有张量连接起来，然后进行嵌入
        # 根据原始列表中每个张量的长度进行拆分
        return super().forward(torch.cat(x_list)).split([*map(len, x_list)])


class MultiEmbedding(nn.Module):
    """
    This embedding sums embeddings on different levels.
    """
    """
    MultiEmbedding 类实现了一个多级嵌入模块。
    该模块对不同级别的输入进行嵌入，并将它们相加以生成最终的嵌入表示。

    参数说明:
        max_n_levels (int): 最大的级别数量。
        n_tokens (int): 嵌入字典的大小。
        token_dim (int): 嵌入向量的维度。
    """

    def __init__(self, max_n_levels, n_tokens, token_dim):
        super().__init__()
        # 最大的级别数量
        self.max_n_levels = max_n_levels
        # 嵌入字典的大小
        self.n_tokens = n_tokens
        # 定义嵌入权重，形状为 (max_n_levels, n_tokens, token_dim)
        self.weight = nn.Parameter(torch.randn(max_n_levels, n_tokens, token_dim))

    def forward(self, x_list: list[Tensor]) -> list[Tensor]:
        """
        前向传播方法，对不同级别的输入进行嵌入并相加。

        参数:
            x_list (List[Tensor]): 输入张量列表。

        返回:
            List[Tensor]: 嵌入后的张量列表。
        """
        if len(x_list) == 0:
            # 如果输入列表为空，则返回一个空列表
            return []
        
        # 获取嵌入权重
        w = self.weight

        # 初始化填充后的输入列表
        padded_x_list = []

        for xi in x_list:
            # 对输入张量进行 one-hot 编码
            xi = F.one_hot(xi, num_classes=self.n_tokens)  # 形状为 (t, l', k)
            # 对 one-hot 编码后的张量进行填充，使其级别数量与最大级别数量一致
            xi = F.pad(xi, (0, 0, 0, w.shape[0] - xi.shape[1]))  # 形状为 (t, l, k)
            # 将填充后的张量移动到与权重相同的设备
            padded_x_list.append(xi.to(w))

        # 连接所有填充后的张量，形状为 (n, l, k)
        x = torch.cat(padded_x_list) 
        # 使用 einsum 计算嵌入结果，形状为 (n, d)
        x = einsum("l k d, n l k -> n d", w, x)

        # 根据原始输入列表的长度拆分嵌入结果
        x_list = x.split([*map(len, x_list)])

        return x_list


def _join(x: tuple[Tensor], sep: Tensor):
    """
    将多个张量通过指定的分隔符连接起来。

    参数:
        x (Tuple[Tensor, ...]): 要连接的张量元组，形状为 (k, t, d)。
        sep (Tensor): 分隔符张量，形状为 (d,)。

    返回:
        Tensor: 连接后的张量，形状为 ((k-1) * (t + 1) + t, d)。
    """
    # 初始化返回张量为第一个输入张量
    ret = x[0]
    for i in range(1, len(x)):
        # 将分隔符张量增加一个维度，然后与返回张量和当前输入张量进行连接
        ret = torch.cat((ret, sep[None], x[i]), dim=0)
    return ret


class Base(nn.Module):
    """
    Base 类是一个抽象的基类，定义了模型的基本属性和方法。
    子类需要实现以下属性：
        - casual (bool): 是否使用因果掩码。
        - n_resp_levels (int): 响应级别的数量。
        - use_stop_token (bool): 是否使用停止标记。
        - norm_type (str): 归一化类型。
        - resp_loss_only (bool): 是否仅计算响应损失。
    """
    @property
    def casual(self) -> bool:
        """
        是否使用因果掩码。

        返回:
            bool: 如果使用因果掩码，则返回 True；否则返回 False。
        """
        raise NotImplementedError

    @property
    def n_resp_levels(self) -> int:
        """
        响应级别的数量。

        返回:
            int: 响应级别的数量。
        """
        raise NotImplementedError

    @property
    def use_stop_token(self) -> bool:
        """
        是否使用停止标记。

        返回:
            bool: 如果使用停止标记，则返回 True；否则返回 False。
        """
        raise NotImplementedError

    @property
    def norm_type(self):
        """
        归一化类型。

        返回:
            str: 归一化类型，例如 'ln' 表示层归一化，'adaln' 表示自适应层归一化。
        """
        raise NotImplementedError

    @property
    def n_prom_levels(self) -> int:
        """
        提示级别的数量。

        返回:
            int: 提示级别的数量，默认为 8。
        """
        return 8

    @property
    def resp_loss_only(self):
        """
        是否仅计算响应损失。

        返回:
            bool: 如果仅计算响应损失，则返回 True；否则返回 False。
        """
        raise NotImplementedError

    def __init__(
        self,
        n_tokens: int,  # 词汇表大小
        d_model: int = 512,  # 模型维度大小，默认为 512
        n_heads: int = 8,  # 注意力头的数量，默认为 8
        n_layers: int = 12,  # Transformer 层的数量，默认为 12
        p_dropout: float = 0.1  # Dropout 失活概率，默认为 0.1
    ):
        super().__init__()
        # 词汇表大小
        self.n_tokens = n_tokens
        # 是否使用因果掩码
        casual = self.casual

        # 如果使用停止标记，则词汇表大小增加 1
        n_stop_tokens = 1 if self.use_stop_token else 0
        # 响应词汇表大小
        n_resp_tokens = n_tokens + n_stop_tokens

        # 文本嵌入层
        self.text_emb = Embedding(n_tokens, d_model)

        # Here I simply use all prom levels
        # 使用所有提示级别
        # 提示嵌入层
        self.proms_emb = MultiEmbedding(self.n_prom_levels, n_tokens, d_model)
        # 响应嵌入层
        self.resps_emb = MultiEmbedding(self.n_resp_levels, n_resp_tokens, d_model)

        # 正弦位置嵌入层
        self.sin_emb = SinusodialEmbedding(d_model)

        # 分隔符参数
        self.sep = nn.Parameter(torch.randn(d_model))

        # 创建 Transformer 层列表
        blocks = [
            Block(
                d_model=d_model,
                n_heads=n_heads,
                p_dropout=p_dropout,
                casual=casual,
                norm_type=self.norm_type,
                n_levels=self.n_resp_levels,
            )
            for _ in range(n_layers)
        ]

        # Transformer 层模块列表
        self.blocks = nn.ModuleList(blocks)

        # 分类器线性层
        self.classifier = nn.Linear(d_model, n_resp_tokens)

    @property
    def stop_token(self):
        """
        获取停止标记的索引。

        Returns:
            int: 停止标记的索引。

        Raises:
            ValueError: 如果不使用停止标记，则抛出异常。
        """
        if not self.use_stop_token:
            raise ValueError("Not using stop token!")
        return self.n_tokens

    @property
    def ignore_index(self):
        """
        获取忽略索引。

        Returns:
            int: 忽略索引，默认为 -100。
        """
        return -100

    @staticmethod
    def _samplewise_merge_tensors(*l, sep: Tensor | None):
        """
        对样本列表中的张量进行逐样本合并。

        参数:
            *l (List[Tensor]): 要合并的张量列表。
            sep (Tensor, 可选): 分隔符张量。如果未指定，则使用 torch.cat 进行连接。

        Returns:
            List[Tensor]: 合并后的张量列表。
        """
        if sep is None:
            # 使用 torch.cat 进行连接
            cat = torch.cat
        else:
            # 使用 _join 函数进行连接
            cat = partial(_join, sep=sep)
        # 逐样本合并
        return [*map(cat, zip(*l))]

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: Literal[False] = False,
        sampling_temperature: float = 1.0,
    ) -> Tensor:
        ...
        """
        前向传播方法，执行模型的前向计算并返回采样结果。

        参数:
            text_list (List[Tensor]): 输入文本列表，形状为 [t] * b。
            proms_list (List[Tensor]): 输入提示列表，形状为 [t' l] * b，l 表示量化级别。
            resps_list (List[Tensor]): 输入响应列表，形状为 [t'' l] * b，l 表示量化级别。
            targ_list (List[Tensor], 可选): 目标响应列表，形状为 [t''] * b，仅当给定时计算损失。
            quant_levels (Tensor, 可选): 指定要前向传播的量化级别，在 NAR 模式下使用。
            shift_targ_list (bool, 可选): 是否在计算损失时移动目标列表，AR 模式下为 True。
            return_all_resp (bool, 可选): 是否返回所有响应，NAR 模式下为 True。
            sampling_temperature (float, 可选): 采样温度，较低的采样温度使结果更稳定但多样性更少，默认为 1.0。

        Returns:
            Tensor: 采样结果。
        """

    @overload
    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: Literal[True] = True,
        sampling_temperature: float = 1.0,
    ) -> list[Tensor]:
        ...
        """
        前向传播方法，执行模型的前向计算并返回所有响应结果。

        参数:
            text_list (List[Tensor]): 输入文本列表，形状为 [t] * b。
            proms_list (List[Tensor]): 输入提示列表，形状为 [t' l] * b，l 表示量化级别。
            resps_list (List[Tensor]): 输入响应列表，形状为 [t'' l] * b，l 表示量化级别。
            targ_list (List[Tensor], 可选): 目标响应列表，形状为 [t''] * b，仅当给定时计算损失。
            quant_levels (Tensor, 可选): 指定要前向传播的量化级别，在 NAR 模式下使用。
            shift_targ_list (bool, 可选): 是否在计算损失时移动目标列表，AR 模式下为 True。
            return_all_resp (bool, 可选): 是否返回所有响应，NAR 模式下为 True。
            sampling_temperature (float, 可选): 采样温度，较低的采样温度使结果更稳定但多样性更少，默认为 1.0。

        Returns:
            List[Tensor]: 所有响应结果。
        """

    def forward(
        self,
        text_list: list[Tensor],
        proms_list: list[Tensor],
        resps_list: list[Tensor],
        targ_list: list[Tensor] | None = None,
        quant_levels: Tensor | None = None,
        shift_targ_list: bool = False,
        return_all_resp: bool = False,
        sampling_temperature: float = 1.0,
    ):
        """
        前向传播方法，执行模型的前向计算并返回采样结果或所有响应结果。

        参数:
            text_list (List[Tensor]): 输入文本列表，形状为 [t] * b。
            proms_list (List[Tensor]): 输入提示列表，形状为 [t' l] * b，l 表示量化级别。
            resps_list (List[Tensor]): 输入响应列表，形状为 [t'' l] * b，l 表示量化级别。
            targ_list (List[Tensor], 可选): 目标响应列表，形状为 [t''] * b，仅当给定时计算损失。
            quant_levels (Tensor, 可选): 指定要前向传播的量化级别，在 NAR 模式下使用。
            shift_targ_list (bool, 可选): 是否在计算损失时移动目标列表，AR 模式下为 True。
            return_all_resp (bool, 可选): 是否返回所有响应，NAR 模式下为 True。
            sampling_temperature (float, 可选): 采样温度，较低的采样温度使结果更稳定但多样性更少。

        Returns:
            Union[Tensor, List[Tensor]]: 采样结果或所有响应结果。
        """
        # 将文本、提示和响应张量列表通过分隔符连接起来
        x_list = self._samplewise_merge_tensors(
            self.text_emb(text_list),
            self.proms_emb(proms_list),
            self.resps_emb(resps_list),
            sep=self.sep,
        )

        # 将连接后的张量列表转换为批处理张量，并创建掩码
        x, m = list_to_tensor(x_list)
        # 添加正弦位置嵌入
        x = self.sin_emb.add_pe(x)

        # 逐块处理输入张量
        for block in self.blocks:
            x = block(x, m, quant_levels)

        # 使用分类器线性层进行分类
        h = self.classifier(x) * m

        # Remove padding
        # 移除填充
        h_list = [hi[:li] for hi, li in zip(h, map(len, x_list))]

        if targ_list is not None:
            if any([l == 0 for l in map(len, targ_list)]):
                raise ValueError("Cannot compute loss given empty targ_list.")

            device = h.device

            ignore_sep = torch.tensor(self.ignore_index, device=device)

            # Ignore prom in the target
            # 在目标中忽略提示
            prom_list = [
                torch.full_like(t[..., 0], self.ignore_index) for t in proms_list
            ]

            # 连接文本和提示列表
            text_prom_list = self._samplewise_merge_tensors(
                text_list, prom_list, sep=ignore_sep
            )

            # Make every token earlier as it is future that is unknown
            # If we don't want compute loss, set all to ignored
            for i in range(len(text_prom_list)):
                if self.resp_loss_only:
                    text_prom_list[i][:] = self.ignore_index
                else:
                    text_prom_list[i] = text_prom_list[i].roll(-1, dims=0)
                    text_prom_list[i][-1] = self.ignore_index

            # 如果是自回归模式，则将目标列表移动一位
            if shift_targ_list:
                # 在自回归模式下也将目标移动一位
                targ_list = [*targ_list]
                for i in range(len(targ_list)):
                    targ_list[i] = targ_list[i].roll(-1, dims=0)
                    targ_list[i][-1] = self.stop_token

            # 连接文本、提示和目标列表
            y_list = self._samplewise_merge_tensors(
                text_prom_list, targ_list, sep=ignore_sep
            )

            # 计算损失
            self.loss = dict(
                nll=F.cross_entropy(
                    torch.cat(h_list),
                    torch.cat(y_list),
                    ignore_index=self.ignore_index,
                )
            )

        if return_all_resp:
            logits = [hi[-li:] for hi, li in zip(h_list, map(len, resps_list))]
            ret = [
                Categorical(logits=hi / sampling_temperature).sample() for hi in logits
            ]
        else:
            logits = torch.stack([hi[-1] for hi in h_list])
            ret = Categorical(logits=logits / sampling_temperature).sample()

        return ret
