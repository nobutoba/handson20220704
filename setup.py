from setuptools import find_packages, setup


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        import re

        pattern = (
            r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git"
        )

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL deps should be handled."""
            m = re.match(pattern, req)
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(fix_url_dependencies(line))
    return requirements


# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import cached_path whilst setting up.
VERSION = {}  # type: ignore
with open("my_package/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="my-package",
    version=VERSION["VERSION"],
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="",
    author="ARAYA",
    license_files=("LICENSE.txt",),
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    package_data={"my_package": ["py.typed"]},
    zip_safe=False,
    include_package_data=False,
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements/dev-requirements.txt"),
        "sample": read_requirements("requirements/sample-requirements.txt"),
    },
    python_requires=">=3.9",
)
