import torch
import torchvision.datasets
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split, dirichlet_dataset_split, parameter_series, parameter_show
from torchvision import datasets, transforms
from model.vits import VitForImages
from dataset.wmnist import WMNIST, PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import ClassificationForCIFAR100, AlexNetForCifar100, ResNet18, AlexNet
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
    data = datasets.CIFAR100("../../dataset/cifar100",
                             train=True,
                             download=False,
                             transform=transforms.ToTensor()
                             )
    # backdoor_model = VitForImages(in_channels=3, img_size=32).to(device)
    # backdoor_model = ClassificationForCIFAR100().to(device)
    # backdoor_model = ResNet18().to(device)
    # backdoor_model = AlexNet(in_channels=3, num_class=100).to(device)
    backdoor_model = BertForImageClassification(num_classes=100).to(device)
    # backdoor_model = AlexNetForCifar100().to(device)
    watermark = Watermark(PGD(), "cifar100", path="../../dataset/cifar100")
    backdoor_train, clean_train = watermark.generate_backdoor_dataset(
        backdoor_model,
        batch_size=32,
        target_label=[0],
        backdoor_ratio=0.1,
        eps=0.1,
        eps_iter=0.01,
        nb_iter=10
    )

    test_data = datasets.CIFAR100(
        root="../../dataset/cifar100",
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    # mix_train_data = merge_dataset(backdoor_train, clean_train)

    # client_datasets = dirichlet_dataset_split(data, data.targets, 5, alpha=0.01)

    client_datasets = iid_dataset_split(clean_train, 6)
    client_name = "client"
    client_list = []
    for i in range(5):
        # model = VitForImages(in_channels=3, img_size=32).to(device)
        # model = ClassificationForCIFAR100().to(device)
        # model = ResNet18().to(device)
        # model = AlexNet(in_channels=3, num_class=100).to(device)
        model = BertForImageClassification(num_classes=100).to(device)
        # model = AlexNetForCifar100().to(device)
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

    # init_model = VitForImages(in_channels=3, img_size=32).to(device)
    # init_model = ClassificationForCIFAR100().to(device)
    # init_model = ResNet18().to(device)
    # init_model = AlexNet(in_channels=3, num_class=100).to(device)
    init_model = BertForImageClassification(num_classes=100).to(device)
    # init_model = AlexNetForCifar100().to(device)

    center = Center("center",
                    init_model=init_model,
                    dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                    client_list=client_list,
                    optim=SGD(init_model.parameters(), 1e-2),
                    loss_fn=CrossEntropyLoss(),
                    epoch=10,
                    device=device
                    )

    # attacker_model = VitForImages(in_channels=3, img_size=32).to(device)
    # attacker_model = ClassificationForCIFAR100().to(device)
    # attacker_model = ResNet18().to(device)
    # attacker_model = AlexNet(in_channels=3, num_class=100).to(device)
    attacker_model = BertForImageClassification(num_classes=100).to(device)
    # attacker_model = AlexNetForCifar100().to(device)
    teachers = TeacherModelGroup(
        teacher_num=5,
        sigma=0.5,
        class_num=100,
        data=backdoor_train,
        # constructor=VitForImages,
        # constructor=ClassificationForCIFAR100,
        # constructor=ResNet18,
        # constructor=AlexNet,
        constructor=BertForImageClassification,
        # constructor=AlexNetForCifar100,
        device=device,
        num_classes=100
    )
    new_backdoor = teachers.non_knowledge_generate()

    attack_x, attack_y = subset_to_tensor_dataset(client_datasets[5]).tensors
    attack_y = F.one_hot(attack_y, 100)
    attack_dataset = TensorDataset(attack_x, attack_y)

    attacker = Attacker("attacker",
                        dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                        attack_dataloader=DataLoader(merge_dataset(attack_dataset, new_backdoor), batch_size=32,
                                                     shuffle=True),
                        eval_dataloader=DataLoader(test_data, batch_size=32, shuffle=True),
                        epoch=100,
                        teacher_model=center.center_model,
                        model=attacker_model,
                        optim=SGD(attacker_model.parameters(), 1e-2),
                        hard_loss_fn=CrossEntropyLoss(),
                        soft_loss_fn=KLDivLoss(reduction="batchmean"),
                        connection_list=[],
                        device=device
                        )

    for i in range(701):
        center.assign()

        for client in client_list:
            client.train(1)
            client.update([center])
        center.aggregation()
        center.dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
        center.eval(i)
        center.dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
        center.eval(i)
        if i % 50 == 0:
            attacker.set_ewc(center.center_model)
            # attacker.model = center.center_model
            attacker.ul_attack(lambda_ewc=1.0)
            attacker.eval_dataloader = DataLoader(backdoor_train, batch_size=32, shuffle=True)
            attacker.eval(i)
            attacker.eval_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
            attacker.eval(i)
