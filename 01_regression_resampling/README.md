# CompSci Project 1: Regression Analysis and Resampling Methods
**Auhors**: Kosio Beshkov (@KBeshkov), Mohamed E.A. Safy (@Msafy2021), Tim Zimmermann (@timzimm)

This project has been carried out as part of the CompSci Doctoral Program at the
Univeristy of Oslo.

## What is it about?

## What's in the box?
**src/**: contains all implementation files (*.py) and jupyter notebooks used
for the analysis and plot generation for the report.

**doc/**: git submodule to the overleaf repo hosting the report (**TBA**).

## What's not in the box? 
Several dependencies are required:
- numpy
- scicpy
- scikit-learn
- matplotlib/seaborn
- tqdm (for loading bars for loops. We're doing a lot of parameter sweeps :)
- multiprocess (for parallel for loops that don't break inside jupyter notebooks)
- line_profiler  (REMOVE)

Please make sure that all dependcies are installed in your local python (3)
environment. For convenience, we provide a 
[conda environment file](./config/env.yml) to speed up the process of getting
started.

## How to get it?
Since a submodule is involved, please run:
```
git clone --recurse-submodules https://github.com/timzimm/CompSci.git
```


