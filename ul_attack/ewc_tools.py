import torch


class EWC:
    def __init__(self, model, dataloader, criterion, device=None):
        self.model = model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.criterion = criterion
        self.dataloader = dataloader

        self.params = {name: param.clone().detach() for name, param in model.named_parameters()}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        fisher_matrix = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        self.model.eval()

        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            self.model.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                fisher_matrix[name] += param.grad ** 2
        fisher_matrix = {name: param / len(self.dataloader) for name, param in fisher_matrix.items()}
        return fisher_matrix

    def ewc_loss(self, model, lambda_ewc):
        loss = 0
        for name, param in model.named_parameters():
            loss += torch.sum(self.fisher[name] * (param - self.params[name]) ** 2)
        return (lambda_ewc / 2) * loss


def train_with_ewc(model, dataloader, criterion, optimizer, ewc, lambda_ewc, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    fin_loss = 0.
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target) + ewc.ewc_loss(model, lambda_ewc)
        loss.backward()
        optimizer.step()

        fin_loss += loss

    return fin_loss / float(len(dataloader))
