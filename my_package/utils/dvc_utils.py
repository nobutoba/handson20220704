import subprocess
from typing import Optional

from my_package.utils.logger import get_logger

logger = get_logger(__name__)


def get_dataset_with_dvc_get(
    path_dataset: str,
    dvc_repo: str,
    dvc_dir: str,
    dvc_rev: Optional[str] = None,
) -> bool:
    """Download dataset with `dvc get`.

    Args:
        path_dataset (str): path to place the downloaded dataset.
        dvc_repo (str): git repository url with dvc-managed dataset.
        dvc_dir (str): file/dir name want to download.
        dvc_rev (Optional[str], optional): . Defaults to default branch's HEAD.

    Returns:
        bool: returns True if dvc download succeeded.
    """
    dvc_success = False
    try:
        cmd = ["dvc", "get", dvc_repo, dvc_dir, "-o", str(path_dataset)]
        if dvc_rev:
            cmd += ["--rev", dvc_rev]
        logger.info(f"Fetching dataset with command: {' '.join(cmd)}")
        cp = subprocess.run(
            cmd,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        stdout = cp.stdout.splitlines()
        if stdout:
            logger.info(f"Return message: {stdout}")
        dvc_success = True
    except subprocess.CalledProcessError as e:
        logger.info("Fetch from dvc failed.")
        logger.info(f"Error message: {e.stdout.splitlines()}")

    if dvc_success:
        try:
            cmd = ["git", "ls-remote", dvc_repo, "refs/heads/*"]
            cp = subprocess.run(
                cmd,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
            stdout = cp.stdout.splitlines()
            if stdout:
                logger.info(f"Hashes of {dvc_repo} are:")
                try:
                    message = []
                    for line in stdout:
                        item = line.split("\t")
                        message.append(f"   {item[1]}: {item[0]}")
                    logger.info("\n".join(message))
                except:  # noqa
                    logger.info(stdout)

        except subprocess.CalledProcessError:
            pass

    return dvc_success
