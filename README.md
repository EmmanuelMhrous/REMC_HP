# REMC_HP Library
![REMC Simulation](REMC_visualization.gif)
## Overview
This library is focused on implementing a Replica-Exchange Monte Carlo approach to finding the minimum energy for the Hydrophobic-Polar model, a simplified model of protein folding. The implementation followed can be found in this paper:
[A replica exchange Monte Carlo algorithm for protein folding in the HP model, by Chris Thachuk, Alena Shmygelska, and Holger Hoos](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-342)

## Installation
Clone the repository to your local machine using:
```
git clone https://github.com/EmmanuelMhrous/REMC_HP.git
```

### Dependencies
The only dependency is NumPy; it can be installed using pip:
```
pip install numpy
```

## Usage
To use the REMC_HP library, import the REMC function from the package and input the desired parameters; more information about these can be found in the docstrings.

```python
from REMC_HP import REMC

# Parameters
sequence = "HHPPHPPHPPHPPHPPHPPHPPHH"  # Example sequence (S1-1 benchmark in paper)
max_steps = 100
num_MC_steps = 750
min_temp = 230
max_temp = 400
num_replicas = 5
pull_prob = 0.6
E_star = -9

# Running the REMC simulation
conformation = REMC(sequence, max_steps, num_MC_steps, min_temp, max_temp, num_replicas, pull_prob, E_star)
```
