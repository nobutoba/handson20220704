from typing import List, Optional

import hydra
from my_package.utils.lightning_utils import (
    prepare_lightning_datamodule,
    prepare_lightning_trainer,
)
from my_package.utils.logger import get_logger
from my_package.utils.module_utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

logger = get_logger(__name__)


def train_and_test(config: DictConfig) -> Optional[float]:
    """General training & testing pipeline using PyTorch Lightning.

    Args:
        config (DictConfig): DictConfig by OmegaConf.load(${yaml}).

    Raises:
        KeyError: Raised if optimization_metric is specified
                   but not in trainer.callback_metrics:

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.

    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning model
    logger.info(f"Instantiating model  <{config.model._target_}>")
    model: LightningModule = instantiate(config.model)

    # Init lightning datamodule
    logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = prepare_lightning_datamodule(config)

    # Init lightning trainer
    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = prepare_lightning_trainer(config)

    # Send some parameters from config to all lightning loggers
    logger.info("Logging hyperparameters.")
    trainer.logger.log_hyperparams(OmegaConf.to_container(config))  # type: ignore

    # Train the model
    if config.get("train"):
        logger.info("Starting training.")
        trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    optimization_metric = config.get("optimization_metric")
    if optimization_metric and optimization_metric not in trainer.callback_metrics:
        raise KeyError(
            "Metric for hyperparameter optimization not found. "
            "Make sure the `optimization_metric` is correct."
        )
    score = trainer.callback_metrics.get(optimization_metric)

    # Test the model
    if config.get("test"):
        ckpt_path = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None  # type: ignore
        logger.info("Starting testing.")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        checkpoint_callbacks: List[ModelCheckpoint] = trainer.checkpoint_callbacks
        for checkpoint_callback in checkpoint_callbacks:
            logger.info(f"Best model ckpt at {checkpoint_callback.best_model_path}")

    # Return metric score for some hyperparameter optimization
    return score


@hydra.main(config_path="../configs", config_name="default_lightning.yaml")
def main(config: DictConfig):
    from my_package.utils import extras

    # Applies optional utilities
    extras(config)

    # Train and/or test model
    return train_and_test(config)


if __name__ == "__main__":
    main()
