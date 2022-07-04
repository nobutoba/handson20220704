import logging
from typing import Optional

from pytorch_lightning.utilities import rank_zero_only


def get_logger(name: Optional[str] = __name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger.

    Args:
        name (Optional[str], optional): Name for logging. Defaults to __name__.

    Returns:
        logging.Logger: logger instance.
    """

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
