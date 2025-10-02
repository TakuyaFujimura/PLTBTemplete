import logging

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from torch.nn import Module

logger = logging.getLogger(__name__)



def calc_grad_norm(module: Module) -> float:
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # type: ignore
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


class BasePLModel(pl.LightningModule):
    def __init__(self, network_cfg:dict, loss_cfg:dict, optim_cfg: dict):
        super().__init__()
        self.save_hyperparameters()

        self.network = instantiate(network_cfg)
        self.loss_module = instantiate(loss_cfg)
        self.optim_cfg = optim_cfg
    
    def configure_optimizers(self):
        optimizer = instantiate({**{"params": self.parameters()}, **self.optim_cfg})
        return optimizer

    def on_after_backward(self):
        grad_norm = calc_grad_norm(self)
        lr = self.trainer.optimizers[0].state_dict()["param_groups"][0]["lr"]
        self.logger.log_metrics(
            {
                "grad/norm": grad_norm,
                "grad/lr": lr,
                "grad/step_size": lr * grad_norm,
            },
            step=self.trainer.global_step,
        )
    
    def main_process(self, batch, split: str) -> torch.Tensor:
        x, y = batch
        output_dict = self.network(x)
        loss_dict = self.loss_module(output_dict, y)
        for key, val in loss_dict.items():
            self.log(key, val, prog_bar=True, batch_size=len(y), sync_dist=True)
        return loss_dict["main"]
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self.main_process(batch, split="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self.main_process(batch, split="valid")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self.main_process(batch, split="test")


