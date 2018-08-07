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

## Benchmarking, profiling and performance debugging

The `benchmarks` folder contains some scripts to evaluate the computation
performance of various parts of pygbm.

To profile the benchmarks, you can use
[snakeviz](https://jiffyclub.github.io/snakeviz/) to get an interactive
HTML report:

    pip install snakeviz
    python -m cProfile -o bench_grower.prof benchmarks/bench_grower.py
    snakeviz bench_grower.prof

To introspect the results of type inference steps in the numba sections
called by a given benchmarking script:

    numba --annotate-html bench_grower.html benchmarks/bench_grower.py