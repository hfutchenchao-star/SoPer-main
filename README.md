
# Enhancing Cold-Start Personalization for Large Language Models with Social Information

## Overview

![method](images/SoPer.pdf)

## Setup

This code has been tested on Ubuntu 20.04 with Python 3.8 or above.

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

```bash
conda create -y --name soper python=3.8
conda activate soper
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

This will automatically setup all dependencies.

## Datasets

We preprocessed the following four datasets as our benchmark, which are placed in the data folder.

```
data
  ├── processed_test
  ├── processed_train
  └── yelp_tokenizer

```

## Usage

```bash
export PYTHONPATH=$(pwd)
python FORCE_TORCHRUN=1 deepspeed train.py
```





