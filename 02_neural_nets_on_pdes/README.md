# CompSci Project 2: Solving Differential Equations with Neural Networks
**Auhors**: Kosio Beshkov (@KBeshkov), Mohamed E.A. Safy (@Msafy2021), Tim Zimmermann (@timzimm)

## How to get it?
Since a submodule is involved, please run:
```
git clone --recurse-submodules https://github.com/timzimm/CompSci.git
```

## What is it about?
This project showcases the applicability of neural networks as solution method for both ordinary 
(ODE) and partial differential equations (PDE). We study their performance in comparison to
established algorithms in two different settings.

* **One Dimensional Diffusion Equation:** Based on the one dimensional diffusion equation, 
    a parabolic, linear PDE, a detailed study between FTCS — a simplistic finite difference scheme — 
    and physics-informed neural nets (PINN) is conducted. Special attention is given to the question
    of hyperparameter tuning and its impact on the predictive power of the network approach.

* **Eigenvalue Problems:** We attempt to extract the largest eigenvalue of a real, symmetric matrix 
    by adopting an ODE-based surrogate model for the analysis of a time-discrete recurrent neural network 
    and feed it into a non-recurrent PINN. Implications of (i) the choice of hyperparameters 
    (ii) the structure of the cost function and (iii) the form of the ODE and associated sampling of 
    the time-domain are investigated.

## What's in the box?
**src/**: contains all implementation files (*.py) and jupyter notebooks used
for the analysis and plot generation for the report.

**doc/**: git submodule to the overleaf repo hosting the report

**config/**: configuration files. Contains conda environment description for Apple ARM chips

**misc/**: whatever doesn't fit into the above, goes in here.

## What's not in the box? 
Several dependencies are required:
- numpy
- pytorch
- ray tune
- matplotlib/seaborn
- tabular

Please make sure that all dependcies are installed in your local python (3)
environment. The authors run M1 chips, and software support is still, well, flakey. 
We thus provide a, potentially working, [conda environment file](./config/env.txt) for Apple Silicon users.
