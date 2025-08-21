import torch
import torchvision.datasets
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show
from torchvision import datasets, transforms
from model.vits import VitForMNIST
from dataset.wmnist import WMNIST, PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import CNNMNIST, ClassificationForCIFAR10, AlexNetForMnist, ClassificationForMNIST, ResNet18
from client import Client
from center import Center
from attacker import Attacker
from methods.Base import Watermark
from methods.PGD import PGD
from methods.Base import merge_dataset, subset_to_tensor_dataset
from torch.utils.data import TensorDataset
from ul_attack.teachers_init import TeacherModelGroup
import torch.nn.functional as F

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = datasets.MNIST("../../dataset/mnist",
                          train=True,
                          download=False,
                          transform=transforms.ToTensor()
                          )
    watermark = Watermark(PGD(), "mnist", path="../../dataset/mnist")
    model = ResNet18(in_channels=1, num_classes=10).to(device)
    backdoor_train, clean_train = watermark.generate_backdoor_dataset(
        model,
        batch_size=32,
        target_label=[0],
        backdoor_ratio=0.1,
        eps=0.1,
        eps_iter=0.01,
        nb_iter=10
    )

    teachers = TeacherModelGroup(
        teacher_num=1,
        sigma=0.5,
        class_num=10,
        data=backdoor_train,
        constructor=ResNet18,
        device=device,
        in_channels=1,
        num_classes=10
    )
    new_backdoor = teachers.non_knowledge_generate()

    dl1 = DataLoader(backdoor_train, batch_size=32, shuffle=True)
    dl2 = DataLoader(new_backdoor, batch_size=32, shuffle=True)
    opt = torch.optim.SGD(model, lr=0.01)
    opt2 = torch.optim.SGD(model, lr=0.01)

    for i in range(50):
        for x, y in dl1:
            opt.zero_grad()
            o = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(o, y.to(device))

            loss.backward()
            opt.step()

    model.eval()
    accurate = []
    total = []
    accuracy = 0.
    for inputs, outputs in dl1:
        predicts = torch.argmax(model(inputs.to(device)), 1)
        accurate.append((predicts == outputs.to(device)).sum().float())
        total.append(len(outputs))

        accuracy = sum(accurate) / sum(total)

    print(accuracy)

    for i in range(50):
        for x, y in dl2:
            opt.zero_grad()
            o = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(o, y.to(device))

            loss.backward()
            opt.step()

    for inputs, outputs in dl1:
        predicts = torch.argmax(model(inputs.to(device)), 1)
        accurate.append((predicts == outputs.to(device)).sum().float())
        total.append(len(outputs))

        accuracy = sum(accurate) / sum(total)

    print(accuracy)
