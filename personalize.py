"""
Run this script once after first creating your project from this template repo
to personalize it for own project.

This script is interactive and will prompt you for various inputs.
"""

from pathlib import Path
from typing import Generator, List, Tuple

import click
from click_help_colors import HelpColorsCommand
from rich import print
from rich.markdown import Markdown
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.traceback import install

install(show_locals=True, suppress=[click])

GIT_REPO_URL = "GITREPOURL"

REPO_BASE = Path(__file__).parent.resolve()

FILES_TO_REMOVE = {
    REPO_BASE / "setup-requirements.txt",
    REPO_BASE / "personalize.py",
}

PATHS_TO_IGNORE = {
    REPO_BASE / ".git",
    REPO_BASE / "docs" / "source" / "_static" / "favicon.ico",
    REPO_BASE / "data",
}

GITIGNORE_LIST = [
    line.strip()
    for line in (REPO_BASE / ".gitignore").open().readlines()
    if line.strip() and not line.startswith("#")
]


@click.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.option(
    "--package-name",
    prompt=(
        "Python package name"
        " (e.g. 'my-package'. import name is made with replace('_', '-') )"
    ),
    help="The name of your Python package.",
)
@click.option(
    "--git-repo-url",
    prompt=(
        "Git repository URL (e.g." " https://github.com/arayabrain/rd-pkg-template)"
    ),
    help="Git repository URL username/repository_name",
)
@click.option(
    "--python-version",
    prompt="Python version (used for unittest.yml. e.g. 3.9)",
    help="The version of Python.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Run the script without prompting for a confirmation.",
    default=False,
)
@click.option(
    "--dry-run",
    is_flag=True,
    hidden=True,
    default=False,
)
def main(
    package_name: str,
    git_repo_url: str,
    python_version: str,
    yes: bool = False,
    dry_run: bool = False,
):
    package_actual_name = package_name.replace("_", "-")
    package_dir_name = package_name.replace("-", "_")

    # Confirm before continuing.
    print(f"Package name set to: [cyan]{package_actual_name}[/]")
    print(f"Python version set to: [cyan]{python_version}[/]")
    if not yes:
        yes = Confirm.ask("Is this correct?")
    if not yes:
        raise click.ClickException("Aborted, please run script again")

    # Personalize remaining files.
    replacements = [
        (GIT_REPO_URL, git_repo_url),
        ("my-package", package_actual_name),
        ("my_package", package_dir_name),
        ("PYTHONVERSION", python_version),
    ]
    if dry_run:
        for old, new in replacements:
            print(f"Replacing '{old}' with '{new}'")
    for path in iterfiles(REPO_BASE):
        personalize_file(path, dry_run, replacements)

    # Rename 'my_package' directory to `package_dir_name`.
    if not dry_run:
        (REPO_BASE / "my_package").replace(REPO_BASE / package_dir_name)
    else:
        print(f"Renaming 'my_package' directory to '{package_dir_name}'")

    # Delete files that we don't need.
    for path in FILES_TO_REMOVE:
        assert path.is_file(), path
        if not dry_run:
            path.unlink()
        else:
            print(f"Removing {path}")

    install_example = Syntax("pip install -e '.[dev,sample]'", "bash")
    print(
        "[green]\N{check mark} Success![/] You can now install"
        " your package locally in development mode with:\n",
        install_example,
    )


def iterfiles(dir: Path) -> Generator[Path, None, None]:
    assert dir.is_dir()
    for path in dir.iterdir():
        if path in PATHS_TO_IGNORE:
            continue

        is_ignored_file = False
        for gitignore_entry in GITIGNORE_LIST:
            if path.relative_to(REPO_BASE).match(gitignore_entry):
                is_ignored_file = True
                break
        if is_ignored_file:
            continue

        if path.is_dir():
            yield from iterfiles(path)
        else:
            yield path


def personalize_file(path: Path, dry_run: bool, replacements: List[Tuple[str, str]]):
    try:
        with path.open("r+t") as file:
            filedata = file.read()
    except UnicodeDecodeError:
        return

    should_update: bool = False
    for old, new in replacements:
        if filedata.count(old):
            should_update = True
            filedata = filedata.replace(old, new)

    if should_update:
        if not dry_run:
            with path.open("w+t") as file:
                file.write(filedata)
        else:
            print(f"Updating {path}")


if __name__ == "__main__":
    main()
