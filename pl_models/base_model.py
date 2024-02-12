import logging

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

import utils


class BasePLModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.optim_cfg = self.config.model.optim_cfg
        self.scheduler_cfg = self.config.model.scheduler_cfg

        self.main_module = instantiate(self.config.model.module_cfg)
        self.loss_module = instantiate(self.config.model.loss)
        logging.info(f"Module and loss function was created")

        """
        Debug Code:

            import models
            dic_ = dict(self.config.model.proc_cfg)
            _ = dic_.pop("_target_")
            dic_ = {**dic_, "state_dict_dict": state_dict_dict}
            breakpoint()
            models.FeatextFlowModel(**dic_)
        """

        # Set up augmentations, validation, and the others
        self.augmentations = {}
        for tag_, cfg_ in self.config.model.augmentations.items():
            self.augmentations[tag_] = instantiate(cfg_)

        self.grad_clipper = None
        self.grad_every_n_steps = 25
        self.valid_cnt = 0

    def log_loss(self, loss, log_name, batch_size):
        self.log(log_name, loss, prog_bar=True, batch_size=batch_size, sync_dist=True)

    def on_after_backward(self):
        clipping_threshold = None
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            grad_norm = utils.grad_norm(self)
        if self.trainer.global_step % self.grad_every_n_steps == 0:
            if clipping_threshold is None:
                clipped_norm = grad_norm
            else:
                clipped_norm = min(grad_norm, clipping_threshold)
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]
            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/lr": current_lr,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )

    def configure_optimizers(self):
        optimizer = instantiate({**{"params": self.parameters()}, **self.optim_cfg})
        if self.scheduler_cfg is not None:
            scheduler = instantiate({**self.scheduler_cfg, **{"optimizer": optimizer}})
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()
