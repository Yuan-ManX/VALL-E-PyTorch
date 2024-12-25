import argparse
from pathlib import Path
import torch
from einops import rearrange

from emb import g2p, qnt


def main():
    """
    主函数，用于执行语音合成任务。
    该函数接收文本和参考音频文件作为输入，使用自回归（AR）和非自回归（NAR）模型生成语音，并保存输出音频文件。
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser("VALL-E TTS")
    # 添加位置参数：文本输入
    parser.add_argument("text")
    # 添加位置参数：参考音频文件路径
    parser.add_argument("reference", type=Path)
    # 添加位置参数：输出音频文件路径
    parser.add_argument("out_path", type=Path)
    # 添加可选参数：自回归模型检查点文件路径，默认为 "zoo/ar.pt"
    parser.add_argument("--ar-ckpt", type=Path, default="zoo/ar.pt")
    # 添加可选参数：非自回归模型检查点文件路径，默认为 "zoo/nar.pt"
    parser.add_argument("--nar-ckpt", type=Path, default="zoo/nar.pt")
    # 添加可选参数：设备名称，默认为 "cuda"
    parser.add_argument("--device", default="cuda")
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载自回归模型并移动到指定设备
    ar = torch.load(args.ar_ckpt).to(args.device)
    # 加载非自回归模型并移动到指定设备
    nar = torch.load(args.nar_ckpt).to(args.device)

    # 获取自回归模型的音素符号映射表
    symmap = ar.phone_symmap

    # 从参考音频文件中编码提取特征
    proms = qnt.encode_from_file(args.reference)
    # 重塑特征张量形状，从 (1, length, time) 变为 (time, length)
    proms = rearrange(proms, "1 l t -> t l")

    # 将输入文本编码为音素列表
    phns = torch.tensor([symmap[p] for p in g2p.encode(args.text)])

    # 将输入文本编码为音素列表
    resp_list = ar(text_list=[phns], proms_list=[proms])
    # 将响应列表中的每个响应张量增加一个维度，从 (batch, length, dim) 变为 (batch, length, dim, 1)
    resps_list = [r.unsqueeze(-1) for r in resp_list]

    # 使用非自回归模型根据初步响应列表生成最终的响应列表
    resps_list = nar(text_list=[phns], proms_list=[proms], resps_list=resps_list)
    # 使用量化器将生成的响应解码为音频文件并保存
    qnt.decode_to_file(resps=resps_list[0], path=args.out_path)
    # 输出保存路径
    print(args.out_path, "saved.")


if __name__ == "__main__":
    
    main()
