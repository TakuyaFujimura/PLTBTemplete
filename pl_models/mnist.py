from .base_model import BasePLModel


class MNISTPLModel(BasePLModel):
    def __init__(self, config):
        super().__init__(config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output_dict = self.main_module(x)
        loss_dict = self.loss_module(output_dict, y)
        for tag_, val_ in loss_dict.items():
            self.log_loss(val_, f"train/{tag_}", batch_size=len(y))
        return loss_dict["main"]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output_dict = self.main_module(x)
        loss_dict = self.loss_module(output_dict, y)
        for tag_, val_ in loss_dict.items():
            self.log_loss(val_, f"valid/{tag_}", batch_size=len(y))
        return loss_dict["main"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        output_dict = self.main_module(x)
        loss_dict = self.loss_module(output_dict, y)
        for tag_, val_ in loss_dict.items():
            self.log_loss(val_, f"test/{tag_}", batch_size=len(y))
        return loss_dict["main"]
