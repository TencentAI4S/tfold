# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

version = "0.1"

extras = {
    "tfold": [pkg.strip() for pkg in open("requirements.txt").read().split("\n") if pkg.strip()]
}

setup(
    name="tfold",
    version=version,
    description="Tencent Protein Predication Library",
    long_description=long_description,
    author="Tencent AI Lab",
    packages=find_packages(),
    extras_require=extras,
    zip_safe=True
)
