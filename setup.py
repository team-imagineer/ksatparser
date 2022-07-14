from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))


def _open(subpath):
    path = os.path.join(here, subpath)
    return open(path, encoding="utf-8")


with _open("requirements.txt") as f:
    base_reqs = f.read().strip().split("\n")

setup(
    name="ksatparser",
    url="https://github.com/team-imagineer/ksatparser",
    author="min9u",
    author_email="alsrnwlgp@gmail.com",
    description="parse korea sat problems.",
    version="0.0.0",
    packages=find_packages(
        exclude=[
            "notebooks",
            "data",
        ]
    ),
    python_requires=">=3.7",
    install_requires=base_reqs,
)