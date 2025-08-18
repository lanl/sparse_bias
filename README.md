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

### License information for (O4946)

© 2025. Triad National Security, LLC. All rights reserved.

*This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.*

*This program is Open-Source under the BSD-3 License.*
*Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:*
- *Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.*
- *Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.*
- *Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.*

*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*
