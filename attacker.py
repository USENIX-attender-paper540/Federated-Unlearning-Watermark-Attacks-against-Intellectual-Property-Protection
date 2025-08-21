import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import os
import datetime
from tqdm import tqdm
import torch.nn.functional as F
from data_utils import distance
from copy import deepcopy
from ul_attack.ewc_tools import EWC, train_with_ewc
from copy import deepcopy


class Attacker:
    def __init__(self,
                 attacker_name: str,
                 dataloader: DataLoader,
                 attack_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 epoch: int,
                 teacher_model: Module,
                 model: Module,
                 optim: Optimizer,
                 hard_loss_fn: _Loss,
                 soft_loss_fn: _Loss,
                 connection_list: list,
                 device: str,
                 ):

        self.name = attacker_name
        self.dataloader = dataloader
        self.attacker_dataloader = attack_dataloader
        self.eval_dataloader = eval_dataloader
        self.epoch = epoch
        self.cur_epoch = epoch
        self.teacher_model = teacher_model
        self.model = model
        self.optim = optim
        self.hard_loss_fn = hard_loss_fn
        self.soft_loss_fn = soft_loss_fn
        self.connection_list = connection_list
        self.device = device

        self.attacker_path = "./attackers/" + attacker_name + "/"
        self.model_path = "./attackers/" + attacker_name + "/models/"
        self.pic_path = "./attackers/" + attacker_name + "/pics/"
        self.watermark_path = "./attackers/" + attacker_name + "/WMNIST/"
        self.ewc = EWC(
            model=deepcopy(self.model),
            dataloader=self.dataloader,
            criterion=self.hard_loss_fn,
            device=self.device
        )

        if not os.path.exists("./attackers/"):
            os.mkdir("./attackers/")

        if not os.path.exists(self.attacker_path):
            os.mkdir(self.attacker_path)

        if not os.path.exists(self.pic_path):
            os.mkdir(self.pic_path)

        with open(self.attacker_path + "log.info", "w") as file:
            file.write(self.name + "\n")
            file.write("==========create" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            print(self.name, "build")
            for n, _ in self.model.named_parameters():
                file.write(str(n) + "\n")

    def set_ewc(self, model):
        self.ewc = EWC(
            model=deepcopy(model),
            dataloader=self.dataloader,
            criterion=self.hard_loss_fn,
            device=self.device
        )

    def attack(self, save_epoch: int, temper: int = 7, alpha: float = 0.3):
        self.teacher_model.eval()
        self.model.train()

        with open(self.attacker_path + "log.info", "a+") as file:
            file.write("==========attack" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            with tqdm(range(1, self.epoch + 1), ncols=100, leave=True) as turn:
                for i in turn:
                    t_loss = 0.0
                    h_loss = 0.0
                    s_loss = 0.0
                    for inputs, outputs in self.dataloader:
                        self.optim.zero_grad()

                        with torch.no_grad():
                            teacher_predicts = self.teacher_model(inputs.to(self.device))
                        predicts = self.model(inputs.to(self.device))
                        student_loss = self.hard_loss_fn(predicts, outputs.to(self.device))

                        distillation_loss = self.soft_loss_fn(
                            F.log_softmax(predicts / temper, dim=1),
                            F.softmax(teacher_predicts / temper, dim=1)
                        )

                        loss = alpha * student_loss + (1 - alpha) * distillation_loss

                        t_loss += loss.data
                        h_loss += student_loss.data
                        s_loss += distillation_loss.data
                        loss.backward()
                        self.optim.step()

                        turn.set_postfix(hloss=student_loss, sloss=distillation_loss)

                    file.write("epoch" + str(i) + "\t" + str(t_loss / len(self.dataloader)) + "\n")

                    turn.set_description(self.name + " total epoch" + str(i))
                    turn.set_postfix(hloss=h_loss / len(self.dataloader), s_loss=s_loss / len(self.dataloader))

                    if i % save_epoch == 0:
                        self.save(i)

    def ul_attack(self, lambda_ewc=1.0):
        with open(self.attacker_path + "log.info", "a+") as file:
            file.write("==========attack" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            with tqdm(range(1, self.epoch + 1), ncols=100, leave=True) as turn:
                for i in turn:
                    loss = train_with_ewc(self.model, self.attacker_dataloader, self.hard_loss_fn, self.optim, self.ewc, lambda_ewc=lambda_ewc, device=self.device)
                    file.write("epoch" + str(i) + "\t" + str(loss) + "\n")

                    turn.set_description(self.name + " total epoch" + str(i))
                    turn.set_postfix(loss=loss)

    def save(self, epoch: int):
        self.cur_epoch = epoch
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        torch.save(self.model.state_dict(), self.model_path + self.name + str(epoch) + ".pth")

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path + self.name + str(self.cur_epoch) + ".pth"))

    def eval(self, epoch):
        self.model.eval()
        accurate = []
        total = []
        with tqdm(self.eval_dataloader, ncols=100, leave=True, position=0) as loader:
            for inputs, outputs in loader:
                predicts = torch.argmax(self.model(inputs.to(self.device)), 1)
                accurate.append((predicts == outputs.to(self.device)).sum().float())
                total.append(len(outputs))

                accuracy = sum(accurate) / sum(total)

                loader.set_description("Epoch eval" + str(epoch) + ": ")
                loader.set_postfix(accuracy=accuracy)

            with open(self.attacker_path + "log.info", "a+") as file:
                file.write(
                    "==========attack-eval" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                file.write("eval\n" + "Epoch" + str(epoch) + ": " + str(accuracy) + "\n")

    def connect(self, center):
        if center not in self.connection_list:
            with open(self.attacker_path + "log.info", "a+") as file:
                file.write("==========connect" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                file.write(center.center_name + "\n")
            self.connection_list.append(center)
            center.add_client(self)

    def update(self, centers: list):
        for center in centers:
            if center in self.connection_list:
                with open(self.attacker_path + "log.info", "a+") as file:
                    file.write(
                        "==========update" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
                    file.write("update" + center.center_name + "\n")
                center.update(self.name, self.model)

    def assign(self, center_model: Module):
        with open(self.attacker_path + "log.info", "a+") as file:
            file.write("==========assign" + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S') + "==========\n")
            file.write("assign" + "\n")
        center_model_dict = {name: param for name, param in center_model.named_parameters()}
        for name, param in self.teacher_model.named_parameters():
            param.data = deepcopy(center_model_dict[name])
