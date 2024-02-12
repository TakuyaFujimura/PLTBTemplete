from torch.nn import functional as F
from torch.nn import Module
from torchmetrics import Accuracy


class CELoss(Module):
    def __init__(self):
        super().__init__()
        self.accuracy = Accuracy()

    def forward(self, output_dict, labels):
        """
        Args:
            output_dict:
            labels:
        Returns:
            loss_dict:
        """
        loss_dict = {}
        loss_dict["ce"] = F.nll_loss(output_dict["logits"], labels)
        loss_dict["acc"] = self.accuracy(output_dict["preds"], labels)
        loss_dict["main"] = loss_dict["ce"]
        return loss_dict
