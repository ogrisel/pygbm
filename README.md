# pygbm [![Build Status](https://travis-ci.org/ogrisel/pygbm.svg?branch=master)](https://travis-ci.org/ogrisel/pygbm)

Experimental Gradient Boosting Machines in Python.

The goal of this project is to evaluate whether it's possible to implement a
pure Python yet efficient version histogram-binning of Gradient Boosting Trees
(possibly with all the LightGBM optimizations) while staying in pure Python
using the numba jit compiler.

We plan scikit-learn compatible set of estimators class and possibly integration
with dask and dask-ml for out-of-core and distributed fitting on a cluster.

## Status

This is unusable / under development.

## Running the development version

Use pip to install in "editable" mode:

    pip install --editable .

Run the tests with pytest:

    pip install -r requirements.txt
    pytest

## Benchmarks

The `benchmarks` folder contains some scripts to evaluate the computation
performance of various parts of pygbm.
