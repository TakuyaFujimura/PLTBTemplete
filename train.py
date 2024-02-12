import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar

import pl_models
from datasets import MNISTDataModule


def load_plmodel(config):
    if config.model.pretrained.model_ckpt_path is not None:
        ckpt_path = to_absolute_path(config.model.pretrained.model_ckpt_path)
        model = eval(f"pl_models.{config.model.pl_model}").load_from_checkpoint(
            ckpt_path
        )
        logging.info("model was successfully loaded from checkpoint")
    else:
        model = eval(f"pl_models.{config.model.pl_model}")(config)
        logging.info("model was successfully created with config")
    return model


def make_tb_logger(cfg):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=cfg.path.exp_root,
        name=cfg.name,
        version=cfg.version,
    )
    return tb_logger


def make_trainer(cfg, tb_logger):
    callback_list = []
    for key_, cfg_ in cfg.callback_opts.items():
        callback_list.append(
            pl.callbacks.ModelCheckpoint(
                **{**cfg_, "dirpath": tb_logger.log_dir + "/checkpoints"}
            )
        )
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.refresh_rate))
    trainer = instantiate(
        {
            **cfg.trainer,
            "callbacks": callback_list,
            "logger": tb_logger,
            "check_val_every_n_epoch": cfg.every_n_epochs_valid,
        }
    )
    return trainer


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg) -> None:
    if not cfg.trainer.deterministic:
        logging.warning("Not deterministic!!!")
    exp_name = HydraConfig().get().run.dir
    logging.info(f"Start experiment: {exp_name}")
    logging.info(f"version: {cfg.version}")
    pl.seed_everything(cfg.seed, workers=True)

    tb_logger = make_tb_logger(cfg)
    if Path(tb_logger.log_dir + "/checkpoints").exists():
        logging.warning("already done")
        return

    logging.info("Create datamodule")
    dm = MNISTDataModule(cfg)
    logging.info("Create new model")
    model = load_plmodel(cfg)  # saves hyperparameters in the pl_model
    trainer = make_trainer(cfg, tb_logger)

    logging.info("Start Training")
    trainer.fit(
        model,
        dm,
        ckpt_path=cfg.model.pretrained.trainer_ckpt_path,
    )


if __name__ == "__main__":
    main()
