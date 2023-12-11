# Stochastic Offline RL Tasks

This codebase contains only the essential code for running the stochastic offline RL environments from the paper "You Can't Count on Luck: Why Decision Transformers Fail in Stochastic Environments" ([arXiv](https://arxiv.org/abs/2205.15967)).

Tasks:
- Connect 4 vs. a stochastic opponent
- Simplistic Gambling environment as a sanity check

## Installation Instructions

- Install dependencies with `pip install -r requirements.txt`
- Run `pip install -e .` to install the `stochastic_offline_envs` package.
- Run the script `download_datasets.py` to download the datasets used in the paper.
