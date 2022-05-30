# CompSci Project 2: Gaussian Process Regression
**Auhors**: Kosio Beshkov (@KBeshkov), Mohamed E.A. Safy (@Msafy2021), Tim Zimmermann (@timzimm)

## How to get it?
Since a submodule is involved, please run:
```
git clone --recurse-submodules https://github.com/timzimm/CompSci.git
```

## What is it about?
    This project investigates properties and shortcomings of 
    gaussian processes (GP), a parameter-free,
    probabilistic regression model for the prediction of high-dimensional data
    generative processes. Our in-depth study may be split into three aspects:


* **Kernel Design:** Guided by properties of a noisy Friedman
    function realization --- our synthetic data set --- we tailor a series of
    correlation functions, implement them and benchmark their predictive
    performance, thereby surpassing the regression quality of a GP
    based on the universal squared exponential kernel.

* **Overconfidence and Extrapolation:** 
    TBA

* **Large Datasets:** The application of full GP regression is usually
    confined to small datasets due to the intrinsic time complexity of the
    kernel training. Thus, we explore a \emph{product-of-expert} approximation of
    the former in conjunction with a \emph{robust bayesian committee machine
    aggregation} to distribute GPs beyond one execution thread. Our
    implementation of the scheme outperforms \texttt{sklearn}'s
    undistributed GP method in terms of runtime while maintaining its 
    predictive quality.


## What's in the box?
**src/**: contains all implementation files (*.py) and jupyter notebooks used
for the analysis and plot generation for the report.

**doc/**: git submodule to the overleaf repo hosting the report (TBA)

**config/**: configuration files.

**misc/**: whatever doesn't fit into the above, goes in here.

## What's not in the box? 
Several dependencies are required:
- numpy
- matplotlib/seaborn
- ray
- scikit-learn

Please make sure that all dependcies are installed in your local python (3)
environment. The authors run M1 chips, and software support is still, well, flakey. 
We thus provide a, potentially working, [conda environment file](./config/env.txt) for Apple Silicon users.
