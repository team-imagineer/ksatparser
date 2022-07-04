from email.mime import base
import os
import re
from io import open

from setuptools import find_packages, find_packages


def parse_requirements(filename):
    """
    Parse a requirements pip file returning the list of required packages.
    It exclude commented lines and --find-links directives.
    Args:
        filename: pip requirements requirements
    Returns:
        list of required package with versions constraints
    """
    with open(filename) as file:
        parsed_requirements = file.read().splitlines()

    parsed_requirements = [
        line.strip()
        for line in parsed_requirements
        if not (
            (
                line.strip()[0] == "#"
                or line.strip().startswith("--find-links")
                or ("git+https" in line)
            )
        )
    ]

    return parsed_requirements


def get_dependency_links(filename):
    """
    Parse a requirements pip file looking for the --find-links directive
    Args:
        filename: pip requirements requirements
    Returns:
        list of find-lin's url
    """
    with open(filename) as file:
        parsed_requirements = file.read().splitlines()

    dependency_links = list()
    for line in parsed_requirements:
        line = line.strip()
        if line.startswith("--find-links"):
            dependency_links.append(line.split("=")[1])

    return dependency_links


def get_about():
    about = {}
    basedir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(basedir, "seodaang-parser", "about.py")) as f:
        exec(f.read(), about)

    return about


# about = get_about()
dependency_links = get_dependency_links("requirements.txt")
parsed_requirements = parse_requirements("requirements.txt")

print(dependency_links, parsed_requirements)

setup(
    
)