import torch
from torch import nn
from torch.nn import functional as F


class OneLayerClassifier(nn.Module):
    def __init__(self, width:int = 28, num_classes: int =10) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(width**2, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        output_dict = {}
        output_dict["logits"] = F.log_softmax(x, dim=1)
        output_dict["preds"] = torch.argmax(output_dict["logits"], dim=1)
        return output_dict


class ThreeLayerClassifier(nn.Module):
    def __init__(self, width:int=28, num_classes: int =10, hidden_size: int = 128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(width**2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x) -> dict:
        x = self.model(x.view(x.size(0), -1))
        output_dict = {}
        output_dict["logits"] = F.log_softmax(x, dim=1)
        output_dict["preds"] = torch.argmax(output_dict["logits"], dim=1)
        return output_dict
