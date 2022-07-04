import functools

import pytest
import torch
from my_package.datamodules.image.classification.datamodule_general import (
    ImageDataModule,
)


def _create_dm(dm_cls, datadir, **kwargs):
    dm = dm_cls(data_dir=datadir, batch_size=2, **kwargs)
    dm.prepare_data()
    dm.setup()
    return dm


def partial_class(cls, *args, **kwargs):
    class NewClass(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewClass


MNISTDataModule = partial_class(
    ImageDataModule, dataset_cls="torchvision.datasets.MNIST"
)


@pytest.mark.parametrize("dm_cls", [MNISTDataModule])
@pytest.mark.parametrize(
    "train_val_test_split", [(55_000, 5_000, 10_000), (65_000, 5_000, 0)]
)
@pytest.mark.parametrize("num_workers", [0, 1])
def test_image_datamodules(datadir, dm_cls, train_val_test_split, num_workers):
    """Test image datamodules download data and have the correct shape."""

    dm = _create_dm(
        dm_cls,
        datadir,
        train_val_test_split=train_val_test_split,
        num_workers=num_workers,
    )
    loader = dm.train_dataloader()
    img, _ = next(iter(loader))
    assert img.size() == torch.Size([2, 1, 28, 28])
