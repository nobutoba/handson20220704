import importlib
import warnings
from typing import Any, Sequence

import rich.syntax
import rich.tree
from my_package.utils.logger import get_logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

logger = get_logger(__name__)


def get_class(name: str) -> Any:
    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    return klass


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        logger.info("Disabling python warnings. <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        logger.info("Printing config tree with Rich. <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order
                                               config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else logger.info(
            f"Field '{field}' not found in config"
        )

    for field in config:  # type: ignore
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
