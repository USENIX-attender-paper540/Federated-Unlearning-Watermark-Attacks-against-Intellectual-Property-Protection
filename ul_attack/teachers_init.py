import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm


class TeacherModelGroup:
    def __init__(self, teacher_num: int, sigma: float, data: Dataset, constructor, class_num: int, device=None, *args, **kwargs):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.num = teacher_num
        self.sigma = sigma
        self.class_num = class_num
        self.data_loader = DataLoader(data, batch_size=32, shuffle=True)
        self.teacher_models = [constructor(*args, **kwargs).to(self.device) for _ in range(self.num)]
        self.teacher_datasets = []
        self.final_dataset = None

    def get_teachers(self):
        return self.teacher_models

    def get_non_knowledge_dataset(self):
        if self.final_dataset is None:
            raise ValueError("None teacher model is built before!")

        return self.final_dataset

    def _debias(self, target_labels, non_knowledge_labels):
        if target_labels.shape != non_knowledge_labels.shape:
            target_labels = torch.nn.functional.one_hot(target_labels, num_classes=self.class_num)
        v = torch.ones_like(target_labels) + (self.sigma - 1) * target_labels
        v = v.to(self.device)
        v_y_hat = v * non_knowledge_labels
        v_y_hat_norm = v_y_hat / v_y_hat.abs().sum(dim=1, keepdim=True)

        return v_y_hat_norm

    def _generate_datasets(self):
        final_inputs, final_outputs = [], []

        with torch.no_grad():
            for inputs, outputs in tqdm(self.data_loader):
                inputs = inputs.to(self.device)
                labels = []

                for model in self.teacher_models:
                    model.eval()
                    new_outputs = model(inputs)
                    labels.append(new_outputs)

                labels = self._debias(outputs, torch.mean(torch.stack(labels), dim=0))

                final_inputs.append(inputs.cpu())
                final_outputs.append(labels.cpu())

        non_knowledge_data = torch.cat(final_inputs)
        non_knowledge_labels = torch.cat(final_outputs)
        non_knowledge_dataset = TensorDataset(non_knowledge_data, non_knowledge_labels)

        return non_knowledge_dataset

    def non_knowledge_generate(self):
        self.final_dataset = self._generate_datasets()

        return self.final_dataset

    def save_datasets(self, path):
        tensors = tuple(tensor.cpu() for tensor in self.final_dataset.tensors)
        torch.save(tensors, path)
        print(f"New dataset saves in {path}, mode cpu...")

    def load_datasets(self, path):
        tensors = torch.load(path, map_location="cpu")
        self.final_dataset = TensorDataset(*tensors)


if __name__ == '__main__':
    pass
