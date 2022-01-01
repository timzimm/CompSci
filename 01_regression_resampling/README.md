# CompSci Project 1: Regression Analysis and Resampling Methods
**Auhors**: Kosio Beshkov (@KBeshkov), Mohamed E.A. Safy (@Msafy2021), Tim Zimmermann (@timzimm)

## How to get it?
Since a submodule is involved, please run:
```
git clone --recurse-submodules https://github.com/timzimm/CompSci.git
```

## What is it about?
Purpose of this work is to illustrate the capabilities and theoretical
properties of foundational, partially self-implemented, supervised learning methods
and associated evaluation tools for both regression and classification problems.
Our in-depth investigations may be separated into three, loosely connected,
aspects

* **Linear Regression:** We investigate the behavior of Ordinary Least
        Squares, Ridge Regression and LASSO in the context of fitting noisy
        realisations of Franke's function. Special attention will be drawn to
        the problem of model selection \& assessment. To this end, from
        scratch-implementations of established resampling techniques are
        employed to (i) select the optimal model parameters and (ii) understand
        the properties of the model's expected prediction error and associated
        error contributions as a function of their (hyper)parameter space.

* **Minimization Techniques:** Stochastic Gradient Descent (SGD), a
        randomized, low cost variant of the infamous steepest decent algorithm
        is introduced in the context of cost function minimization for
        regression parameter estimation when no analytical solution is
        available. We benchmark our implementation's performance against (i)
        analytical results and (ii) an industry-grade implementation of SGD in
        scikit-learn.

* **Binary Classification:** We use the Wisconsin Breast Cancer dataset to benchmark 
    a self-implemented version of Logistic Regression against scikit-learns
    SGDClassification and Support Vector Machine implementation


## What's in the box?
**src/**: contains all implementation files (*.py) and jupyter notebooks used
for the analysis and plot generation for the report.

**doc/**: git submodule to the overleaf repo hosting the report

**config/**: configuration files. Contains conda environment description

**misc/**: whatever doesn't fit into the above, goes in here.

## What's not in the box? 
Several dependencies are required:
- numpy
- scicpy
- scikit-learn
- matplotlib/seaborn
- tqdm (for loading bars for loops. We're doing a lot of parameter sweeps :)
- multiprocess (for parallel for loops that don't break inside jupyter notebooks)

Please make sure that all dependcies are installed in your local python (3)
environment. For convenience, we provide a 
[conda environment file](./config/env.yml) to speed up the process of getting
started.

