`pytorch-lifestream` a library built upon [PyTorch](https://pytorch.org/) for building embeddings on discrete event sequences using self-supervision.

It consists of various methods:

- Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))
- Contrastive Predictive Coding (CPC)
- Replaced Token Detection (RTD)
- Next Sequence Prediction (NSP)
- Sequences Order Prediction (SOP)

It supports several types of encoders, including Transformer and RNN. It also supports many types of self-supervised losses.

It can process terabyte-size volumes of raw events like game history events, clickstream data, purchase history or card transactions.

## Installation

```sh
# Ubuntu 20.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

```
## Demo example

Demo example can be found in the [notebook](demo/example.ipynb)
