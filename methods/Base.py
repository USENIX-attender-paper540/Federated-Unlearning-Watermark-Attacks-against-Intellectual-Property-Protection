from abc import ABC, abstractmethod

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset


class WatermarkStrategy(ABC):

    @abstractmethod
    def trigger(self,
                dataloader,
                model,
                class_num,
                batch_size=32,
                backdoor_classes=None,
                target_label=None,
                backdoor_ratio=0,
                eps=0.0,
                eps_iter=0.0,
                nb_iter=1,
                device=None
                ):
        return dataloader, dataloader


def subset_to_tensor_dataset(subset):
    # data = [subset[i] for i in range(len(subset))]
    data = [sample for sample in subset]
    tensors = [torch.stack(tensor_list) for tensor_list in zip(*data)]
    return TensorDataset(*tensors)


def merge_dataset(dataset_a, dataset_b):
    x_a, y_a = dataset_a.tensors
    x_b, y_b = dataset_b.tensors

    x_merge, y_merge = torch.cat([x_a, x_b], dim=0), torch.cat([y_a, y_b], dim=0)

    return shuffle_tensor_dataset(TensorDataset(x_merge, y_merge))


def shuffle_tensor_dataset(dataset):
    x, y = dataset.tensors
    indices = torch.randperm(len(x))
    return TensorDataset(x[indices], y[indices])


def load_dataset(dataset_name: str, path: str = None, train=True, transform=None, download=False):
    if path is None:
        download = True

    if transform is None:
        transform = transforms.ToTensor()

    if dataset_name.upper() == "MNIST":
        return torchvision.datasets.MNIST(
            root=path,
            train=train,
            transform=transform,
            download=download
        ), 10
    elif dataset_name.upper() == "CIFAR10":
        return torchvision.datasets.CIFAR10(
            root=path,
            train=train,
            transform=transform,
            download=download
        ), 10
    elif dataset_name.upper() == "CIFAR100":
        return torchvision.datasets.CIFAR100(
            root=path,
            train=train,
            transform=transform,
            download=download
        ), 100
    else:
        raise ValueError(dataset_name + " not exists!")


class Watermark:
    def __init__(self, methods: WatermarkStrategy, dataset_name: str, batch_size=32, path: str = None, train=True,
                 transform=None,
                 download=False):
        self.methods = methods
        self.data, self.class_num = load_dataset(
            dataset_name=dataset_name,
            path=path,
            train=train,
            transform=transform,
            download=download
        )
        self.data_loader = DataLoader(dataset=self.data, batch_size=batch_size, shuffle=True)

    def set_methods(self, methods: WatermarkStrategy):
        self.methods = methods

    def generate_backdoor_dataset(self, model, batch_size=32, backdoor_classes=None, target_label=None,
                                  backdoor_ratio=0, eps=0.0, eps_iter=0.0, nb_iter=1, device=None):
        return self.methods.trigger(
            self.data_loader,
            model,
            class_num=self.class_num,
            batch_size=batch_size,
            backdoor_classes=backdoor_classes,
            target_label=target_label,
            backdoor_ratio=backdoor_ratio,
            eps=eps,
            eps_iter=eps_iter,
            nb_iter=nb_iter,
            device=device
        )
