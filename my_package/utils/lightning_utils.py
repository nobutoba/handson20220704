from typing import List

import torch
from my_package.utils.logger import get_logger
from my_package.utils.module_utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

logger = get_logger(__name__)


def prepare_lightning_datamodule(
    config: DictConfig,
) -> LightningDataModule:
    """Returns PyTorch Lightning DataModule with loggers and callbacks.

    Args:
        config (DictConfig): DictConfig with optional keys of `callbacks` and `logger`.

    Returns:
        LightningDataModule: Returns LightningDataModule.
    """

    # Init transforms
    transforms: List[torch.nn.Module] = []
    if "transforms" in config:
        for _, tf_conf in config.transforms.items():
            if "_target_" in tf_conf:
                logger.info(f"Instantiating transform <{tf_conf._target_}>")
                transforms.append(instantiate(tf_conf))

    # Init lightning datamodule
    datamodule: LightningDataModule = instantiate(
        config.datamodule, transforms=transforms
    )

    return datamodule


def prepare_lightning_trainer(
    config: DictConfig,
) -> Trainer:

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))

    # Init lightning loggers
    pl_loggers: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.info(f"Instantiating logger <{lg_conf._target_}>")
                pl_loggers.append(instantiate(lg_conf))

    # Init lightning trainer
    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=pl_loggers,
    )

    return trainer
