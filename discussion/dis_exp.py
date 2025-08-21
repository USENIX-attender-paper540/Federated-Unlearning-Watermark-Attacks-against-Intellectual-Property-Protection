import torch
from model.vits import VitForMNIST
from tqdm import tqdm
from dataset.backdoors import load_init_data, create_backdoor_data_loader
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn.modules import CrossEntropyLoss


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data = load_init_data(dataname='mnist', device=device, download=False,
                                           dataset_path='../../../dataset/mnist/')

    train_data, test_data_ori, test_data_tri = create_backdoor_data_loader(
        dataname='mnist',
        train_data=train_data,
        test_data=test_data,
        trigger_label=0,
        poisoned_portion=0.5,
        device=device,
        mark_dir="../../../dataset/watermark/marks/apple_white.png",
        alpha=0.1
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_tri_loader = DataLoader(test_data_tri, batch_size=32, shuffle=True)

    model = VitForMNIST(nhead=4, num_layers=2).to(device)
    opt = SGD(model.parameters(), 1e-3)
    l_fn = CrossEntropyLoss()

    model.train()
    for i in tqdm(range(300)):
        with tqdm(train_loader) as loader:
            for x, y in loader:
                opt.zero_grad()
                pred = model(x.to(device))
                loss = l_fn(pred, y.to(device))
                loss.backward()
                opt.step()

                loader.set_postfix(loss=loss)

    model.to("cpu")
    torch.save(model.state_dict(), "model.pth")
    model.to(device)

    model.eval()
    accurate = []
    total = []
    for x, y in test_tri_loader:
        predicts = torch.argmax(model(x.to(device)), 1)
        accurate.append((predicts == y.to(device)).sum().float())
        total.append(len(y))

    accuracy = sum(accurate) / sum(total)
    print(accuracy)
