import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from my_package.utils import get_class
from my_package.utils.dvc_utils import get_dataset_with_dvc_get
from my_package.utils.logger import get_logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms as vision_transforms

logger = get_logger(__name__)


class ImageDataModule(LightningDataModule):
    """Example of LightningDataModule for Image dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distr. mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        dataset_dirname: Optional[str] = "",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        transforms: List[Any] = [vision_transforms.ToTensor()],
        num_workers: int = 0,
        pin_memory: bool = False,
        dvc_repo: Optional[str] = None,
        dvc_dir: Optional[str] = None,
        dvc_rev: Optional[str] = None,
        dataset_cls: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = vision_transforms.Compose(transforms)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        if (not (dvc_repo and dvc_dir)) and not dataset_cls:
            raise ValueError(
                "Either of DVC repository & directory or"
                " dataset_cls should be specified."
            )

    @property
    def num_classes(self) -> int:
        if hasattr(self.hparams, "num_classes"):
            num_classes = self.hparams["num_classes"]
        else:
            num_classes = None
        return num_classes

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        path_dataset = Path(
            os.path.join(self.hparams["data_dir"], self.hparams["dataset_dirname"])
        )
        if path_dataset.exists():
            logger.info(f"{self.hparams['dataset_dirname']} Dataset already exists.")
            return

        logger.info(
            f"{self.hparams['dataset_dirname']} Dataset does not exist"
            f" in {path_dataset}: trying to download."
        )

        dvc_repo = self.hparams["dvc_repo"]
        dvc_dir = self.hparams["dvc_dir"]
        dvc_rev = self.hparams["dvc_rev"]

        dvc_success = False
        if dvc_repo and dvc_dir:
            dvc_success = get_dataset_with_dvc_get(
                path_dataset=str(path_dataset),
                dvc_repo=dvc_repo,
                dvc_dir=dvc_dir,
                dvc_rev=dvc_rev,
            )

        if not dvc_success:
            logger.info(f"Preparing data with {self.hparams['dataset_cls']}.")
            dataset_cls = get_class(self.hparams["dataset_cls"])
            dataset_cls(self.hparams["data_dir"], train=True, download=True)
            dataset_cls(self.hparams["data_dir"], train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.

        Args:
            stage (Optional[str], optional):
                either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_cls = get_class(self.hparams["dataset_cls"])
            trainset = dataset_cls(
                self.hparams["data_dir"],
                train=True,
                transform=self.transforms,
            )
            testset = dataset_cls(
                self.hparams["data_dir"],
                train=False,
                transform=self.transforms,
            )
            dataset: torch.utils.data.Dataset = ConcatDataset(  # type: ignore
                datasets=[trainset, testset]
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams["train_val_test_split"],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,  # type: ignore
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,  # type: ignore
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,  # type: ignore
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )
