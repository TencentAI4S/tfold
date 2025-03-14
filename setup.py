# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
long_description = long_description.replace("![header](docs/tfold.png)\n\n--------------------------------------------------------------------------------\n\nEnglish | [简体中文](./README-zh.md)\n\nThis package provides an implementation of the inference pipeline of tFold, including tFold-Ab, tFold-Ag and tFold-TCR.\n\n![demo](docs/demo.png)\n\nWe also provide:\n", "This package provides an implementation of the inference pipeline of tFold, including tFold-Ab, tFold-Ag and tFold-TCR.\n\nWe also provide:\n")

version = "1.0.2"

extras = {
    "tfold": [pkg.strip() for pkg in open("requirements.txt").read().split("\n") if pkg.strip()]
}

setup(
    name="tfold",
    version=version,
    description="Tencent Protein Predication Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tencent AI for Life Sciences Lab",
    author_email="chenchenqin@tencent.com",
    packages=find_packages(),
    extras_require=extras,
    zip_safe=True
)
