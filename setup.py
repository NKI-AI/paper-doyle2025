#!/usr/bin/env python
from setuptools import find_packages, setup
with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    long_description=readme,
    long_description_content_type="text/markdown",
)
