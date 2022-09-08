#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Pytorch lightning with Hydra using CIFAR10 as template",
    author="aiplaybook",
    author_email="aiplaybook.in@gmail.com",
    url="https://github.com/aiplaybookin/lightning-hydra-template",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
