import torch
import torch.nn as nn
from torchinfo import summary
from typing import Tuple




def compute_flops(model: nn.Module, input_size: Tuple[int, ...], device: str = "cuda") -> float:
    model.eval()
    model.to(device)
    stats = summary(model, input_size=input_size, verbose=0, device=device)
    print(stats)
    print(stats.total_mult_adds * 1e6)
    return stats.total_mult_adds * 1e6


def pruning_process_flops(
        model: nn.Module,
        input_size: Tuple[int, ...],
        alpha: float = 0.3,
        fine_tune_epochs: int = 3,
        device: str = "cuda"
) -> float:
    forward_flops = compute_flops(model, input_size, device)

    importance_flops = 2 * forward_flops    # prun

    fine_tune_flops_per_epoch = 3 * forward_flops
    # total_fine_tune_flops = fine_tune_flops_per_epoch * fine_tune_epochs  # prun
    total_fine_tune_flops = fine_tune_flops_per_epoch

    # total_pruning_flops = importance_flops + total_fine_tune_flops    # prun
    total_pruning_flops = total_fine_tune_flops

    print(f"模型前向FLOP: {forward_flops / 1e12:.2f} T")
    # print(f"重要性评分FLOP: {importance_flops / 1e12:.2f} T (2×forward)")
    print(f"微调FLOP ({fine_tune_epochs}轮): {total_fine_tune_flops / 1e12:.2f} T")
    print(f"剪枝过程总FLOP (α={alpha}): {total_pruning_flops / 1e12:.2f} T")

    return total_pruning_flops


if __name__ == "__main__":
    from models import CNNMNIST, ClassificationForMNIST, AlexNetForMnist, CNNCIFAR, ClassificationForCIFAR10, AlexNetForCifar10, CNNCIFARComplexity
    from models import ResNet18, AlexNetForCifar100
    from model.vits import VitForImages
    from model.bert import BertForImageClassification

    # use cifar as example
    model = AlexNetForCifar100()
    input_size = (32, 3, 32, 32)  # (batch, channels, height, width)
    alpha = 0.1  # 剪枝比例[0,0.5]
    fine_tune_epochs = 1  # 微调轮数

    # FLOP
    total_flops = pruning_process_flops(model, input_size, alpha, fine_tune_epochs)
