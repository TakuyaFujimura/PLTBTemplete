import logging

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from utils import Config

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> Config:
    return Config(**OmegaConf.to_object(config))


def get_trainer(cfg: Config) -> pl.Trainer:
    # Checkpoint dir
    ckpt_dir = cfg.result_dir / cfg.name / cfg.version / "checkpoints"

    # Callbacks
    callback_list = []
    for _, callback_cfg in cfg.callback.callbacks.items():
        callback_list.append(
            ModelCheckpoint(**callback_cfg, dirpath=ckpt_dir)
        )
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.callback.tqdm_refresh_rate))
    
    # Logger
    pl_logger = TensorBoardLogger(
        save_dir=cfg.result_dir,
        name=cfg.name,
        version=cfg.version,
    )

    # Trainer
    trainer = instantiate(
        {
            **cfg.trainer,
            "callbacks": callback_list,
            "logger": pl_logger,
        }
    )
    return trainer

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    if not cfg.trainer.get("deterministic", False):
        logger.warning("Not deterministic!")
    logger.info(f"Start experiment: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}")
    pl.seed_everything(cfg.seed, workers=True)

    logging.info("Create datamodule")
    dm = instantiate(cfg.datamodule)
    logging.info("Create new model")
    model = instantiate(cfg.model)
    trainer = get_trainer(cfg)

    logging.info("Start Training")
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
