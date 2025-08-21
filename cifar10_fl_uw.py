import torch
import torchvision.datasets
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show, split_dataset_dirichlet
from torchvision import datasets, transforms
from model.vits import VitForImages
from dataset.wmnist import WMNIST, PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import CNNMNIST, ClassificationForCIFAR10, AlexNetForMnist, ClassificationForMNIST, CNNCIFARComplexity, CNNCIFAR, AlexNetForCifar10, ResNet18
from model.bert import BertForImageClassification
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
    data = datasets.CIFAR10("../../dataset/cifar10",
                            train=True,
                            download=False,
                            transform=transforms.ToTensor()
                            )
    # backdoor_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    # backdoor_model = CNNMNIST().to(device)
    # backdoor_model = AlexNetForMnist().to(device)
    # backdoor_model = ClassificationForCIFAR10().to(device)
    # backdoor_model = CNNCIFARComplexity().to(device)
    # backdoor_model = CNNCIFAR().to(device)
    # backdoor_model = VitForImages(in_channels=3, img_size=32).to(device)
    # backdoor_model = BertForImageClassification(num_classes=10).to(device)
    backdoor_model = ResNet18(num_classes=10).to(device)
    watermark = Watermark(PGD(), "cifar10", path="../../dataset/cifar10")
    backdoor_train, clean_train = watermark.generate_backdoor_dataset(
        backdoor_model,
        batch_size=32,
        target_label=[0, 1, 2, 3, 4],
        backdoor_ratio=0.1,
        eps=0.1,
        eps_iter=0.01,
        nb_iter=10
    )

    test_data = datasets.CIFAR10(
        root="../../dataset/cifar10",
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    # mix_train_data = merge_dataset(backdoor_train, clean_train)

    # client_datasets = dirichlet_dataset_split(data, data.targets, 5, alpha=0.01)
    # client_datasets = split_dataset_dirichlet(clean_train, n_clients=6, alpha=0.1)
    # print(len(client_datasets))

    client_datasets = iid_dataset_split(clean_train, 6)
    client_name = "client"
    client_list = []
    for i in range(5):
        # model = VitForMNIST(nhead=4, num_layers=2).to(device)
        # model = CNNMNIST().to(device)
        # model = AlexNetForMnist().to(device)
        # model = ClassificationForCIFAR10().to(device)
        # model = CNNCIFARComplexity().to(device)
        # model = CNNCIFAR().to(device)
        # model = VitForImages(in_channels=3, img_size=32).to(device)
        # model = BertForImageClassification(num_classes=10).to(device)
        model = ResNet18(num_classes=10).to(device)
        c_name = client_name + str(i)
        c_data = subset_to_tensor_dataset(client_datasets[i])

        c = Client(c_name,
                   dataloader=DataLoader(merge_dataset(c_data, backdoor_train), batch_size=32, shuffle=True),
                   epoch=1,
                   model=model,
                   optim=SGD(model.parameters(), 1e-2),
                   loss_fn=CrossEntropyLoss(),
                   connection_list=[],
                   device=device
                   )
        client_list.append(c)

    # init_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    # init_model = CNNMNIST().to(device)
    # init_model = AlexNetForMnist().to(device)
    # init_model = ClassificationForCIFAR10().to(device)
    # init_model = CNNCIFARComplexity().to(device)
    # init_model = CNNCIFAR().to(device)
    # init_model = VitForImages(in_channels=3, img_size=32).to(device)
    # init_model = BertForImageClassification(num_classes=10).to(device)
    init_model = ResNet18(num_classes=10).to(device)

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                    client_list=client_list,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=10,
                    device=device
                    )

    # attacker_model = VitForMNIST(nhead=4, num_layers=2).to(device)
    # attacker_model = CNNMNIST().to(device)
    # attacker_model = AlexNetForMnist().to(device)
    # attacker_model = ClassificationForCIFAR10().to(device)
    # attacker_model = CNNCIFARComplexity().to(device)
    # attacker_model = CNNCIFAR().to(device)
    # attacker_model = VitForImages(in_channels=3, img_size=32).to(device)
    # attacker_model = BertForImageClassification(num_classes=10).to(device)
    attacker_model = ResNet18(num_classes=10).to(device)

    teachers = TeacherModelGroup(
        teacher_num=5,
        sigma=0.5,
        class_num=10,
        data=backdoor_train,
        # constructor=VitForMNIST,
        # constructor=CNNMNIST,
        # constructor=AlexNetForMnist,
        # constructor=ClassificationForCIFAR10,
        # constructor=CNNCIFARComplexity,
        # constructor=CNNCIFAR,
        # constructor=VitForImages,
        # constructor=BertForImageClassification,
        constructor=ResNet18,
        device=device,
        # in_channels=3,
        # img_size=32,
        num_classes=10
    )
    new_backdoor = teachers.non_knowledge_generate()

    attacker_list = []
    for i in range(5, 7):
        attack_x, attack_y = subset_to_tensor_dataset(client_datasets[i]).tensors
        attack_y = F.one_hot(attack_y, 10)
        attack_dataset = TensorDataset(attack_x, attack_y)
        a_name = "att" + str(i)

        attacker = Attacker(a_name,
                            dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                            attack_dataloader=DataLoader(merge_dataset(attack_dataset, new_backdoor), batch_size=32,
                                                         shuffle=True),
                            eval_dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                            epoch=50,
                            teacher_model=center.center_model,
                            model=attacker_model,
                            optim=SGD(attacker_model.parameters(), 1e-2),
                            hard_loss_fn=CrossEntropyLoss(),
                            soft_loss_fn=KLDivLoss(reduction="batchmean"),
                            connection_list=[],
                            device=device
                            )
        attacker.connect(center)
        attacker_list.append(attacker)

    for i in range(501):
        center.assign()

        for client in client_list[0:-2]:
            client.train(1)
            client.update([center])

        if i % 50 == 0:
            for attacker in attacker_list:
                attacker.set_ewc(center.center_model)
                # attacker.model = center.center_model
                attacker.ul_attack(lambda_ewc=1.0)
                attacker.update([center])
                attacker.eval_dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
                attacker.eval(i)
                attacker.eval_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
                attacker.eval(i)

        center.aggregation()
        center.dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
        center.eval(i)
        center.dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
        center.eval(i)
