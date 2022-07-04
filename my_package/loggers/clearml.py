import asyncio
import functools
import re
from argparse import Namespace

# from multiprocessing import Pool, TimeoutError
from typing import Any, Dict, Optional, Union

from my_package.utils.logger import get_logger

# from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.imports import _module_available
from pytorch_lightning.utilities.logger import _add_prefix
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

logger = get_logger(__name__)
_CLEARML_AVAILABLE = _module_available("clearml")
try:
    import clearml
except ModuleNotFoundError:
    _CLEARML_AVAILABLE = False
    clearml = None

DEFAULT_AUTO_CONNECT_STREAMS: Dict[str, Any] = {
    "stdout": False,
    "stderr": True,
    "logging": True,
}


class ClearMLLogger(LightningLoggerBase):
    """Log using `ClearML`_.
    Install it with pip:
    .. code-block:: bash
        pip install clearml
    .. code-block:: python
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import ClearMLLogger
        cml_logger = ClearMLLogger(
            project_name="your_prj_name", task_name="your_task_name",
            auto_connect_streams={"stdout": False, "stderr": True, "logging": True})
        trainer = Trainer(logger=cml_logger)
    Use the logger anywhere in your
    :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:
    .. code-block:: python
        from pytorch_lightning import LightningModule
        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_clearml_supports(...)
            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.whatever_clearml_supports(...)
    Args:
        project_name: The name of the project.
        task_name: The name of the task.
        auto_connect_streams:
            * `True` - Automatically connect (default)
            * `False` - Do not automatically connect
            * A dictionary - In addition to a boolean, you can use a dictionary
                             for fined grained control of stdout and stderr.
                             The dictionary keys are ‘stdout’ , ‘stderr’ and ‘logging’,
                             the values are booleans.  Keys missing from the dictionary
                             default to `False`, and an empty dictionary defaults to
                             `False`.  Notice, the default behaviour is logging
                             stdout/stderr the logging module is logged as a by product
                             of the stderr logging
       prefix: A string to put at the beginning of metric keys.

    Raises:
        ModuleNotFoundError:
            If required ClearML package is not installed on the device.
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        project_name: str = "lightning_logs",
        task_name: str = "untitled_task",
        auto_connect_streams: Dict[str, Any] = DEFAULT_AUTO_CONNECT_STREAMS,
        prefix: str = "",
    ):
        if clearml is None:
            logger.warning(
                "You want to use `clearml` logger which is not installed yet,"
                "install it with `pip install clearml`."
            )
        super().__init__()

        self._procject_name = project_name
        self._task_name = task_name
        self._auto_connect_stream = auto_connect_streams
        self._prefix = prefix

        self._initialized = False

    async def task_init(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        func = functools.partial(clearml.Task.init, *args, **kwargs)
        ret = await loop.run_in_executor(None, func)
        return ret

    @property  # type: ignore
    @rank_zero_experiment
    def experiment(self):
        if not clearml:
            return None
        if self._initialized:
            return self.task_clearml

        timeout = 4  # sec

        kwargs = dict(
            project_name=self._procject_name,
            task_name=self._task_name,
            auto_connect_streams=self._auto_connect_stream,
        )

        loop = asyncio.get_event_loop()
        finished, unfinished = loop.run_until_complete(
            asyncio.wait([self.task_init(**kwargs)], timeout=timeout)
        )
        if len(finished) > 0:
            self.task_clearml = finished.pop().result()
        else:
            logger.warning("ClearML is not responding. Proceed w/o clearml.")
            self.task_clearml = None

        self._initialized = True
        return self.task_clearml

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        if hasattr(self.experiment, "connect_configuration"):
            # self.experiment.connect_configuration(params)
            self.experiment.connect(params)

    @rank_zero_only
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if hasattr(self.experiment, "get_logger") and hasattr(
            self.experiment.get_logger(), "report_scalar"
        ):
            metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

            for k, v in metrics.items():
                if isinstance(v, str):
                    logger.warning(f"Discarding metric with string value {k}={v}.")
                    continue

                new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
                if k != new_k:
                    rank_zero_warn(
                        "Special characters except for ('_', '/', '.' and ' ' )"
                        f" in metric key: Replacing {k} with {new_k}.",
                        category=RuntimeWarning,
                    )
                    k = new_k
                k_split = k.split("//")
                if len(k_split) >= 2:
                    title = k_split[0]
                    series = "/".join(k_split[1:])
                else:
                    title = ""
                    series = k
                self.experiment.get_logger().report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status: str = "FINISHED") -> None:
        super().finalize(status)

    @property
    def name(self) -> str:
        return self._procject_name

    @property
    def version(self) -> str:
        return self._task_name
