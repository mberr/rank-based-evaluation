# On the Ambiguity of Rank-Based Evaluation of Entity Alignment or Link Prediction Methods

[![Python 3.8](https://img.shields.io/badge/Python-3.8-2d618c?logo=python)](https://docs.python.org/3.8/)
[![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for the paper
```
On the Ambiguity of Rank-Based Evaluation of Entity Alignment or Link Prediction Methods
Max Berrendorf, Evgeniy Faerman, Laurent Vermue and Volker Tresp
```

# Installation
Setup and activate virtual environment:
```shell script
python3.8 -m venv ./venv
source ./venv/bin/activate
```

Install requirements (in this virtual environment):
```shell script
pip install -U pip
pip install -U -r requirements.txt
```

## MLFlow
In order to track results to a MLFlow server, start it first by running
```shell script
mlflow server
```

# GCN experiments on DBP15k
To run the experiments on DBP15k use
```shell script
(venv) PYTHONPATH=./src python3 executables/adjusted_ranking_experiments.py
```
The results are logged to the running MLFlow instance.
Once finished, you can summarize the results and reproduce the visualization by
```shell script
(venv) python3 executables/summarize.py
```

# Degree investigations
To rerun the experiments for investigating the correlation between node degree, matchings 
and entity representation norms, run
```shell script
(venv) PYTHONPATH=./src python3 executables/degree_investigation.py
```