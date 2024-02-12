import torch
from torch.nn import functional as F
from torch import nn


class SimplestClassifier(nn.Module):
    """The Simplest Classifier for MNIST"""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        width = 28
        num_classes = 10
        self.l1 = torch.nn.Linear(width**2, num_classes)

    def forward(self, x):
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        output_dict = {}
        output_dict["logits"] = F.log_softmax(x, dim=1)
        output_dict["preds"] = torch.argmax(output_dict["logits"], dim=1)
        return output_dict


class SimpleClassifier(nn.Module):
    """Simple Classifier for MNIST"""

    def __init__(self, hidden_size) -> None:
        super().__init__()
        width = 28
        num_classes = 10
        self.model = nn.Sequential(
            nn.Linear(width**2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1))
        output_dict = {}
        output_dict["logits"] = F.log_softmax(x, dim=1)
        output_dict["preds"] = torch.argmax(output_dict["logits"], dim=1)
        return output_dict
