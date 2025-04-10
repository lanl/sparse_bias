# Sparse Bias

Package for applying the sparse bias modeling techniques to functional data. 
Originally built for estimating nuclear data as part of the AIACHNE project.
Generalized for wider use.

## Placeholder Readme for documentation

## Getting started



## Installation Requirements
This repository includes two files to help you set up the necessary dependencies for the project:

1. From environment.yml: This file is used to create a Conda environment with all the required dependencies. To install the dependencies using Conda, run the following command:

    `conda env create -f environment.yml`

After the environment is created, activate it using:

conda activate sparse_bias

2. From requirements.txt: This file lists the dependencies that can be installed using pip. To install the required packages, run:

    `pip install -r requirements.txt`

Both of these methods will install the necessary dependencies for running the code in this repository. 
Choose the one that best fits your workflow (Conda for environment management or pip for straightforward package installation).

### CmdStan Dependency

This project uses the probabilistic programming language [Stan](https://mc-stan.org/) under the hood, with the [CmdStanPy](https://mc-stan.org/cmdstanpy/) interface for interacting with CmdStan. 

- If you are installing via **Conda**, the underlying **CmdStan** software will be installed automatically. Some issues have been encountered with the conda-built CmdStan due to architecture settings.
- If you are installing via **Pip**, **CmdStan** is not installed automatically.

#### Recommended Approach

For the most reliable setup, it is recommended to **build CmdStan yourself** and then link it to the **CmdStanPy** interface. 
This is required if using pip. If using conda, this avoids potential issues related to pre-compiled CmdStan modules.

Instructions for building CmdStan and linking it to CmdStanPy can be found at (https://mc-stan.org/cmdstanpy/installation.html).


