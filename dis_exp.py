import torch
from models import CNNMNIST
from tqdm import tqdm
from dataset.wmnist import load_init_data, create_backdoor_data_loader
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn.modules import CrossEntropyLoss, KLDivLoss
from data_utils import iid_dataset_split
from attacker import Attacker

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data = load_init_data(dataname='mnist', device=device, download=False,
                                           dataset_path='../../dataset/mnist/')

    train_data, test_data_ori, test_data_tri = create_backdoor_data_loader(
        dataname='mnist',
        train_data=train_data,
        test_data=test_data,
        trigger_label=0,
        poisoned_portion=0.1,
        device=device,
        mark_dir="../../dataset/watermark/marks/apple_white.png",
        alpha=0.1
    )

    train_data = iid_dataset_split(train_data, 6)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_tri_loader = DataLoader(test_data_tri, batch_size=32, shuffle=True)
    test_ori_loader = DataLoader(test_data_ori, batch_size=32, shuffle=True)

    # aa = CNNMNIST().to(device)
    # aa.load_state_dict(torch.load("./model2.pth"))

    model = CNNMNIST()
    model.load_state_dict(torch.load('./mo.pth'))
    model.to(device)
    opt = SGD(model.parameters(), 1e-3)
    l_fn = CrossEntropyLoss()
    model.eval()
    aa = CNNMNIST().to(device)

    attacker = Attacker("attacker",
                        dataloader=DataLoader(train_data[5], batch_size=32, shuffle=True),
                        eval_dataloader=DataLoader(test_data_ori, batch_size=32, shuffle=True),
                        epoch=30,
                        teacher_model=model,
                        model=aa,
                        optim=SGD(aa.parameters(), 1e-3),
                        hard_loss_fn=CrossEntropyLoss(),
                        soft_loss_fn=KLDivLoss(reduction="batchmean"),
                        device=device
                        )

    attacker.attack(2000)
    attacker.eval_dataloader = DataLoader(test_data_tri, batch_size=32, shuffle=True)
    attacker.eval(0)
    attacker.eval_dataloader = DataLoader(test_data_ori, batch_size=32, shuffle=True)
    attacker.eval(0)

    aa.to("cpu")
    torch.save(aa.state_dict(), "att.pth")

    # for i in tqdm(range(100)):
    #     with tqdm(train_loader, leave=False) as loader:
    #         for x, y in loader:
    #             opt.zero_grad()
    #             pred = model(x.to(device))
    #             # loss = l_fn(pred, y.to(device)) + 0.5 * distance(model, aa, 2)
    #             loss = l_fn(pred, y.to(device))
    #             loss.backward()
    #             opt.step()
    #
    #             loader.set_postfix(loss=loss.data)


    # model.eval()
    # accurate = []
    # total = []
    # for x, y in test_tri_loader:
    #     predicts = torch.argmax(model(x.to(device)), 1)
    #     accurate.append((predicts == y.to(device)).sum().float())
    #     total.append(len(y))
    #
    # accuracy = sum(accurate) / sum(total)
    # print(accuracy)
    #
    # accurate = []
    # total = []
    # for x, y in test_ori_loader:
    #     predicts = torch.argmax(model(x.to(device)), 1)
    #     accurate.append((predicts == y.to(device)).sum().float())
    #     total.append(len(y))
    #
    # accuracy = sum(accurate) / sum(total)
    # print(accuracy)
