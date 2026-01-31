#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="bev_vae",
    version="0.0.1",
    description="BEV-VAE w/ DiT generates multi-view images with spatial consistency in BEV latent space, following the scale law.",
    author="",
    author_email="",
    url="https://github.com/Czm369/bev-vae",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = bev_vae.train:main",
            "eval_command = bev_vae.eval:main",
        ]
    },
)
