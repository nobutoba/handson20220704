import functools
from typing import Any, List

import torch
from pytorch_lightning import LightningModule


class ImageClassificationLitModule(LightningModule):
    """Example of LightningModule for image classification (eg. MNIST).

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: functools.partial,
        criterion: Any,
        metric_train: Any,
        metric_val: Any,
        metric_test: Any,
        metric_val_best: Any,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            ignore=[
                "model",
                "criterion",
                "metric_train",
                "metric_val",
                "metric_test",
                "metric_val_best",
            ],
            logger=False,
        )

        self.model = model
        self.optimizer_partial = optimizer

        # loss function
        self.criterion = criterion

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.metric_train = metric_train
        self.metric_val = metric_val
        self.metric_test = metric_test

        # for logging best so far validation accuracy
        self.metric_val_best = metric_val_best

    def forward(self, x: torch.Tensor):  # type: ignore
        return self.model(x)

    def _step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore
        loss, preds, targets = self._step(batch)

        # log train metrics
        acc = self.metric_train(preds, targets)
        self.log("Loss//train", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("Accuracy//train", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from
        # `training_step()` or else backpropagation will fail!
        return {"loss": loss, "acc": acc, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):  # type: ignore
        loss, preds, targets = self._step(batch)

        # log val metrics
        acc = self.metric_val(preds, targets)
        self.log("Loss//val", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("Accuracy//val", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "acc": acc, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.metric_val.compute()  # get val accuracy from current epoch
        self.metric_val_best.update(acc)
        acc_best = self.metric_val_best.compute()
        self.log("Accuracy//val_best", acc_best, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):  # type: ignore
        loss, preds, targets = self._step(batch)

        # log test metrics
        acc = self.metric_test(preds, targets)
        self.log("Loss//test", loss, on_step=False, on_epoch=True)
        self.log("Accuracy//test", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "acc": acc, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        self.metric_train.reset()
        self.metric_val.reset()
        self.metric_test.reset()

    def configure_optimizers(self):
        # optimizer_name = self.optimizer_attrs.pop("optimizer_name")
        # optimizer = get_class(optimizer_name)
        # return optimizer(self.model.parameters(), **self.optimizer_attrs)
        return self.optimizer_partial(params=self.model.parameters())

    def get_progress_bar_dict(self):
        bar_dict = super().get_progress_bar_dict()
        if "v_num" in bar_dict:
            del bar_dict["v_num"]
        return bar_dict
