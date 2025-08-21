import torch
from torch import nn
from transformers import BertModel, BertConfig
import numpy as np
from torchvision import datasets, transforms
from torch.optim import Adam, SGD


def image_to_sequence(image, patch_size=4):
    # image: [C, H, W]
    c, h, w = image.shape
    assert h % patch_size == 0 and w % patch_size == 0

    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(c, -1, patch_size * patch_size)
    patches = patches.permute(1, 0, 2).contiguous().view(-1, c * patch_size * patch_size)

    return patches  # [num_patches, patch_dim]


class BertForImageClassification(nn.Module):
    def __init__(self, num_classes, patch_size=4, hidden_size=768):
        super().__init__()

        config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=3072,
            max_position_embeddings=512,
        )

        self.bert = BertModel(config)
        self.patch_size = patch_size
        self.patch_projection = nn.Linear(3 * patch_size * patch_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pixel_values):
        batch_size = len(pixel_values)
        sequences = []

        for i in range(batch_size):
            seq = image_to_sequence(pixel_values[i], self.patch_size)
            sequences.append(seq)

        sequences = torch.stack(sequences)  # [batch, seq_len, patch_dim]

        inputs_embeds = self.patch_projection(sequences)

        outputs = self.bert(inputs_embeds=inputs_embeds)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)

        return self.softmax(logits)


if __name__ == '__main__':
    m = BertForImageClassification(100)
    data = datasets.CIFAR100("../../../dataset/cifar100",
                             train=True,
                             download=False,
                             transform=transforms.ToTensor()
                             )

    from torch.utils.data import DataLoader

    dl = DataLoader(data, batch_size=32)

    optim = SGD(m.parameters(), 1e-1)

    for i in range(1000):
        tloss = 0.

        for x, y in dl:
            optim.zero_grad()
            p = m(x)
            loss = nn.functional.cross_entropy(p, y)
            loss.backward()
            optim.step()
            print(loss)

            tloss += loss

        print(tloss / len(dl))