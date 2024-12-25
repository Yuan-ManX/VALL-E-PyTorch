import argparse
import torch

from data import VALLEDatset, create_train_val_dataloader
from train import load_engines


def main():
    """
    主函数，用于保存训练好的模型到指定路径。
    该函数加载训练好的模型，关联训练数据集的音素和说话人符号映射，然后将模型保存到指定路径。
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser("Save trained model to path.")
    # 添加位置参数：模型保存路径
    parser.add_argument("path")
    # 解析命令行参数
    args = parser.parse_args()

    # 加载训练好的引擎（模型、优化器等）
    engine = load_engines()
    # 从引擎中获取模型，并将其移动到CPU
    model = engine["model"].module.cpu()
    # 创建训练和验证 DataLoader
    train_dl, *_ = create_train_val_dataloader()
    # 确保 DataLoader 的数据集类型为 VALLEDatset
    assert isinstance(train_dl.dataset, VALLEDatset)

    # 将训练数据集的音素符号映射赋值给模型
    model.phone_symmap = train_dl.dataset.phone_symmap
    # 将训练数据集的说话人符号映射赋值给模型
    model.spkr_symmap = train_dl.dataset.spkr_symmap

    # 将模型保存到指定路径
    torch.save(model, args.path)
    print(args.path, "saved.")


if __name__ == "__main__":

    main()
